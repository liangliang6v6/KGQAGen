import sys
import os
import time
import logging
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

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

# use relation path z to get the reasoning path
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

# use LLM to predict
def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id_ = data["id"]
    # print("@@@@@using model:", model)
    if id_ in processed_list:
        return None
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id_,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    input_q = input_builder.process_input(data)
    prediction = model.generate_sentence(input_q)
    if prediction is None:
        return None
    result = {
        "id": id_,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input_q,
    }
    print("@@@input@@@@:\n",input_q)
    return result


def main(args, LLM):
    # input_file = os.path.join(args.data_path, args.d)
    # dataset = load_dataset(input_file, split=args.split)
    rule_postfix = "no_rule"
    # Load dataset
    input_file = "data/QoG.jsonl"
    print("path!!!!!",input_file)
    dataset = load_dataset(
        "json",
        data_files=input_file,
        split="train" 
    )
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        # use build_qa_input.py 
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    # # calculate the inference time:
    # sum_time = 0.0

    # log_file = f"inference_time.log"
    # log_path = os.path.join(output_dir, log_file)
    
    # logging.basicConfig(
    #     filename=log_path,
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    # )
    # logging.info("Starting inference process...")
    # logging.info("Arguments:")
    # for key, value in vars(args).items():
    #     logging.info(f"  {key}: {value}")
    
    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            # # start inference
            # start_time = time.time()
            res = prediction(data, processed_list, input_builder, model)
            # end_time = time.time()
            # infer_time = end_time - start_time
            # sum_time += infer_time
            # logging.info(f"Sample ID: {data['id']}, Inference Time: {infer_time:.8f} seconds")
            
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()
    # avg_time = sum_time / len(dataset)
    # logging.info(f"Total inference time: {sum_time:.8f} seconds")
    # logging.info(f"Average inference time per sample: {avg_time:.8f} seconds")

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="QoG")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="llama2-chat-hf", #gpt-3.5-turbo
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true", default=False, help="Enable rule-based reasoning (default: True)")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results1/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    #!!!not the RoG didn't call finetuned llama2 here
    # argparser.add_argument(
    #     "--model_path",
    #     type=str,
    #     help="model_name for save results",
    #     default="/data/liang/workspace/KG/train/wiki", #meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct
    # )

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
