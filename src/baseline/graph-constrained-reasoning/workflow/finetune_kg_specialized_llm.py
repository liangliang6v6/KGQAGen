import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch.utils
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
import datasets
from src import utils

datasets.disable_progress_bar()
import dotenv
from accelerate import Accelerator

dotenv.load_dotenv()

PATH_START_TOKEN = "<PATH>"
PATH_END_TOKEN = "</PATH>"

HF_TOKEN = os.getenv("HF_TOKEN")
N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)

ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
{question}
# Topic entities: 
{entities}
"""

ANS_TEMPLATE = """# Reasoning Path:
{reasoning_path}
# Answer:
{answer}"""

@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    n_path_per_sample: int = field(
        default=10, metadata={"help": "Number of paths to sample"}
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit"})
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "attn implementation"})
    response_template: Optional[str] = field(default="[/INST]", metadata={"help": "Response template"})


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)


def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation=script_args.attn_implementation,
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
        # device=torch.device("cuda:1"),
        # device_map={"": Accelerator().local_process_index},
    )

    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        token=HF_TOKEN,
    )

    # Add new tokens
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token # tokenizer.unk_token for LLAMA-2-7b-chat-hf
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    # Add new tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    special_tokens_dict['additional_special_tokens'] = [PATH_START_TOKEN, PATH_END_TOKEN]
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    data_list = [
        datasets.load_from_disk(data_path) for data_path in script_args.data_path_list
    ]
    dataset = datasets.concatenate_datasets(data_list)

    def input_formatter(example):
        chunks = []
        for i in range(len(example["proof"][0][0])):
            question = example["question"][i]
            start_node = example["proof"][0][0][i]
            anser_node = example["answer"][i]
            ground_paths = example["ground_truth_paths"][i]
            if not question.endswith("?"):
                question += "?"
            raw_input = ZERO_SHOT_PROMPT.format(
                question=question, entities=",".join(start_node)
            )
            # Split ground paths into multiple samples
            if len(ground_paths) > 0:
                for path in ground_paths:
                    if len(path) == 0:
                        continue
                    
                    ground_path_string= f"{PATH_START_TOKEN}{utils.path_to_string(path)}{PATH_END_TOKEN}"
                    print("INPUT PATH!!!", ground_path_string)
                    # the last entity in the path is always the answer
                    path_answer = path[-1][-1].strip()
                    response = ANS_TEMPLATE.format(
                        reasoning_path=ground_path_string, answer=path_answer
                    )
                    chat = [
                        {"role": "user", "content": raw_input},
                        {"role": "assistant", "content": response},
                    ]
                    final_input = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=False
                    )
                    chunks.append(final_input)
        return {"text": chunks}

    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )
    
    print(train_dataset[0])
    
    # Prepare instruct tuning
    response_template = script_args.response_template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )
    sft_cfg = SFTConfig(
        **training_args.to_dict(),
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_cfg,
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
