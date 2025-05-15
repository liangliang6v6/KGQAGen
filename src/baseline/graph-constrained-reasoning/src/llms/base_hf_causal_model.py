from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftConfig

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model_path", type=str, help="HUGGING FACE MODEL or model path"
        )
        parser.add_argument("--maximun_token", type=int, help="max length", default=4096)
        parser.add_argument(
            "--max_new_tokens", type=int, help="max length", default=1024
        )
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        parser.add_argument(
            "--attn_implementation",
            default="flash_attention_2",
            choices=["eager", "sdpa", "flash_attention_2"],
            help="enable flash attention 2",
        )
        parser.add_argument(
            "--generation_mode",
            type=str,
            default="greedy",
            choices=["greedy", "beam", "sampling", "group-beam", "beam-early-stopping", "group-beam-early-stopping"],
        )
        parser.add_argument(
            "--k", type=int, default=1, help="number of paths to generate"
        )
        parser.add_argument("--chat_model", default='true', type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--use_assistant_model", default='false', type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--assistant_model_path", type=str, help="HUGGING FACE MODEL or model path", default=None)

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token

    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, token=HF_TOKEN, trust_remote_code=True
        )
        print("@@@load from model:",self.args.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=self.DTYPE.get(self.args.dtype, None),
            load_in_8bit=self.args.quant == "8bit",
            load_in_4bit=self.args.quant == "4bit",
            trust_remote_code=True,
            attn_implementation=self.args.attn_implementation,
        )
        if self.args.use_assistant_model:
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.args.assistant_model_path,
                device_map="auto",
                token=HF_TOKEN,
                torch_dtype=self.DTYPE.get(self.args.dtype, None),
                load_in_8bit=self.args.quant == "8bit",
                load_in_4bit=self.args.quant == "4bit",
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )
        else:
            self.assistant_model = None

        self.maximun_token = self.tokenizer.model_max_length
        try:
            self.generation_cfg = GenerationConfig.from_pretrained(self.args.model_path)
        except:
            # Load from PeftModel
            sft_peft_config = PeftConfig.from_pretrained(self.args.model_path)
            self.generation_cfg = GenerationConfig.from_pretrained(sft_peft_config.base_model_name_or_path)
            
        self.generation_cfg.max_new_tokens = self.args.max_new_tokens
        self.generation_cfg.return_dict_in_generate = (True,)

        if self.args.generation_mode == "greedy":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_return_sequences = 1
        elif self.args.generation_mode == "sampling":
            self.generation_cfg.do_sample = True
            self.generation_cfg.num_return_sequences = self.args.k
        elif self.args.generation_mode == "beam":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = self.args.k
            self.generation_cfg.num_return_sequences = self.args.k
        elif self.args.generation_mode == "beam-early-stopping":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = self.args.k
            self.generation_cfg.num_return_sequences = self.args.k
            self.generation_cfg.early_stopping = True
        elif self.args.generation_mode == "group-beam":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = self.args.k
            self.generation_cfg.num_return_sequences = self.args.k
            self.generation_cfg.num_beam_groups = self.args.k
            self.generation_cfg.diversity_penalty = 1.
        elif self.args.generation_mode == "group-beam-early-stopping":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = self.args.k
            self.generation_cfg.num_return_sequences = self.args.k
            self.generation_cfg.num_beam_groups = self.args.k
            self.generation_cfg.early_stopping = True
            self.generation_cfg.diversity_penalty = 1.

    def prepare_model_prompt(self, query):
        if self.args.chat_model:
            chat_query = [
                {"role": "user", "content": query}
            ]
            return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)
        else:
            return query
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input, *args, **kwargs):
        # outputs = self.generator(
        #     llm_input,
        #     return_full_text=False,
        #     max_new_tokens=self.args.max_new_tokens,
        #     handle_long_generation="hole",
        #     generation_config=self.generation_cfg,
        #     assistant_model = self.assistant_model
        # )
        # return outputs[0]["generated_text"].strip()  # type: ignore
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        try:
            res = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                generation_config=self.generation_cfg,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(e)
            return None
        response = []
        if len(res.sequences) == 1:
            return self.tokenizer.decode(res.sequences[0][input_ids.shape[1]:],skip_special_tokens=True)
        else:
            for r in res.sequences:
                response.append(self.tokenizer.decode(r[input_ids.shape[1]:], 
            skip_special_tokens=True))
            return response
