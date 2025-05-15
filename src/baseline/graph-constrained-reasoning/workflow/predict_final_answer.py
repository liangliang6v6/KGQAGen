import hashlib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from tqdm import tqdm
from src.llms import get_registed_model
import os
from datasets import load_dataset
from src.utils.qa_utils import eval_result
import json
from multiprocessing.dummy import Pool
from functools import partial
from src.qa_prompt_builder import PromptBuilder
from src.utils.graph_utils import build_graph, get_truth_paths
from src.utils import path_to_string, load_jsonl, dfs
from src.trie import MarisaTrie

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def make_prediction(data, args, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    input = input_builder.process_input(data)
    input = model.prepare_model_prompt(input)
    if "gcr" in args.model_name:
        entity_list = set()
        predicted_paths = data['predicted_paths']
        for p in predicted_paths:
            path_list = p.split(" -> ")
            for i in range(0, len(path_list), 2):
                entity_list.add(path_list[i])
        tokenized_entity_list = model.tokenizer(list(entity_list), padding=False, add_special_tokens=False).input_ids
        tokenized_entity_list = [
                    ids + [model.tokenizer.eos_token_id] for ids in tokenized_entity_list
                ]
        entity_trie = MarisaTrie(tokenized_entity_list)
        prediction = model.generate_sentence(input, entity_trie)
    else:
        prediction = model.generate_sentence(input)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
    }
    return result

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def merge_path_result(
    qa_dataset, path_dataset, n_proc=1, filter_empty=False, remove_dup_path=False
):
    question_to_path = dict()
    for data in path_dataset:
        qid = data["id"]
        predicted_paths = (
            list(set(data["prediction"])) if remove_dup_path else data["prediction"]
        )
        ground_paths = data["ground_truth_paths"]
        question_to_path[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_path(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        if qid in question_to_path:
            sample["predicted_paths"] = question_to_path[qid]["predicted_paths"]
            sample["ground_paths"] = question_to_path[qid]["ground_paths"]
        else:
            g = build_graph(sample["graph"])
            start_node = sample["q_entity"]
            answer_node = sample["a_entity"]
            truth_paths = get_truth_paths(start_node, answer_node, g)
            sample["ground_paths"] = [path_to_string(path) for path in truth_paths]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_path, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def get_all_paths(qa_dataset, length = 2, n_proc=1, filter_empty=False):
    def find_path(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        g = build_graph(sample["graph"])
        start_node = sample["q_entity"]
        answer_node = sample["a_entity"]
        all_paths = dfs(g, start_node, max_length=length)
        truth_paths = get_truth_paths(start_node, answer_node, g)
        sample["predicted_paths"] = [path_to_string(path) for path in all_paths]
        sample["ground_paths"] = [path_to_string(path) for path in truth_paths]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_path, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset
    

def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    print(dataset)
    # my_id = "WebQTest-743_bef14a2c5e7b8f9ce875fe024cacdf9c"
    # for row in dataset:
    #     if row["id"] == my_id:
    #         print(row["question"],row["answer"])

    if args.add_path:
        if args.use_all:
            max_length = 2
            prediction_suffix = f"add_all_path_{max_length}"
            dataset = get_all_paths(
                dataset,
                length=max_length,
                filter_empty=args.filter_empty,
            )
        else:
            path_name = "_".join(args.reasoning_path.split("/")[:-1])
            # Prevent too long path name
            if len(path_name) > 64:
                path_name_md5 = hashlib.md5(path_name.encode()).hexdigest()
                path_name = path_name[:64] + "_" + path_name_md5
            prediction_suffix = f"add_path_{path_name}"
            paths_datasets = []
            with open(args.reasoning_path, "r") as f:
                for line in f:
                    paths_datasets.append(json.loads(line))
            dataset = merge_path_result(
                dataset,
                paths_datasets,
                filter_empty=args.filter_empty,
                remove_dup_path=args.remove_dup_path,
            )
        if args.filter_empty:
            prediction_suffix += "_filter_empty"
        if args.use_true:
            prediction_suffix = "add_true_path"
        if args.remove_dup_path:
            prediction_suffix += "_no_dup"
        if  "use_assistant_model" in vars(args) and args.use_assistant_model:
            if args.assistant_model_path is None:
                raise ValueError("Assistant model path is None")
            assistant_model_name = args.assistant_model_path.split("/")[-1]
            if len(assistant_model_name) > 64:
                assistant_model_name_md5 = hashlib.md5(assistant_model_name.encode()).hexdigest()
                assistant_model_name = assistant_model_name[:64] + "_" + assistant_model_name_md5
            prediction_suffix += f"_assistant_{assistant_model_name}"
    elif args.add_rule:
        prediction_suffix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
    else:
        prediction_suffix = "no-path"
    if args.use_rog_prompt:
        prediction_suffix += "_rog_prompt"
    if "gcr" in args.model_name:
        prediction_suffix += f"_k_{args.k}_gen_{args.generation_mode}"
    prediction_suffix = args.prefix + prediction_suffix
    
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, prediction_suffix
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)

    input_builder = PromptBuilder(
        add_rule=args.add_rule,
        add_path=args.add_path,
        use_true=args.use_true,
        maximun_token=model.maximun_token,
        tokenize=model.token_len,
        use_rog_prompt=args.use_rog_prompt,
        each_line=args.each_line,
    )

    print("Prepare pipline for inference...")
    print(args)
    print(model)
    model.prepare_for_inference()
    

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    fout, processed_list = get_output_file(
        os.path.join(output_dir, f"predictions.jsonl"), force=args.force
    )

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(p.imap(
                partial(
                    make_prediction,
                    args=args,
                    processed_list=processed_list,
                    input_builder=input_builder,
                    model=model,
                ),
                dataset,
            ), total=len(dataset)
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = make_prediction(data, args, processed_list, input_builder, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(os.path.join(output_dir, f"predictions.jsonl"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="lianglz")
    argparser.add_argument("--d", "-d", type=str, default="QoG")
    argparser.add_argument("--split", type=str, default="test")#test[:100]
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument("--prefix", type=str, default="")
    argparser.add_argument(
        "--reasoning_path",
        type=str,
        default="results/GenPaths/RoG-webqsp/gcr-Llama-2-7b-chat-hf_ft-flash-neftune-RoG-webqsp-cwq-lora-False-3-epochs/test[:100]/zero-shot-group-beam-k30/predictions.jsonl",
    )
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="/home/lluo/projects/LLMRuleQA/results/gen_rule_path/webqsp/Llama-2-7b-chat-hf_align-spectoken-joint-explainqa/test/predictions_3_False.jsonl",
    )
    argparser.add_argument("--add_rule", type=lambda x: (str(x).lower() == "true"), default=False)
    argparser.add_argument(
        "--add_path", type=lambda x: (str(x).lower() == "true"), default=False
    )
    argparser.add_argument(
        "--model_name", type=str, help="model_name for save results", default="llama3"
    )
    argparser.add_argument(
        "--force", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument(
        "--debug", action="store_true", help="print debug information"
    )
    argparser.add_argument(
        "--use_true", type=lambda x: (str(x).lower() == "true"), default=False
    )
    argparser.add_argument(
        "--use_all", type=lambda x: (str(x).lower() == "true"), default=False
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument(
        "--filter_empty", type=lambda x: (str(x).lower() == "true"), default=False
    )
    argparser.add_argument(
        "--remove_dup_path", type=lambda x: (str(x).lower() == "true"), default=True
    )
    argparser.add_argument(
        "--use_rog_prompt", type=lambda x: (str(x).lower() == "true"), default=False
    )
    argparser.add_argument(
        "--each_line", type=lambda x: (str(x).lower() == "true"), default=True
    )
    argparser.add_argument(
        '--use_gcr', type=lambda x: (str(x).lower() == 'true'), default=False
    )
    args, _ = argparser.parse_known_args()

    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)

    args = argparser.parse_args()

    main(args, LLM)
