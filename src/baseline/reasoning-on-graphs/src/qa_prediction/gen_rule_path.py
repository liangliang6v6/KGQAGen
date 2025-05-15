import json
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import utils
from datasets import load_dataset
import datasets

datasets.disable_progress_bar()
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import re
from dotenv import load_dotenv

load_dotenv()
N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
PATH_RE = re.compile(r"<PATH>(.*?)</PATH>")
INSTRUCTION="""Please generate a valid wikidata based relation path that can be helpful for answering the following question: """

# INSTRUCTION="""Please generate the shortest possible relation path that directly connects entities and can answer the following question. Ensure the relation path includes only the most relevant relationships to avoid unnecessary complexity."""
# # hugging face token
HF_TOKEN = os.getenv("HF_TOKEN")

# read the file
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

# get the relation path z, use <SEP> to replace the entity
INVALID_SUBSTR = re.compile(r"(?:<unk>|</s>|[`>{}\[\]])")
VALID_REL   = re.compile(r"^[A-Za-z][A-Za-z0-9 _:]+$")

def parse_prediction(predictions: list[str]) -> list[list[str]]:
    """
    Parse raw paths into cleaned relation‐lists. Beams with at least one valid
    relation survive; we drop only individual junk relations.
    """
    results = []
    for p in predictions:
        p = p.strip()
        if not p:
            continue

        # 1) extract inside tags if present
        m = PATH_RE.search(p)
        block = m.group(1).strip() if m else p

        # 2) split on <SEP>
        parts = block.split("<SEP>")
        cleaned = []
        for rel in parts:
            # 3a) strip whitespace & punctuation
            rel = rel.strip()
            rel = re.sub(r"(</?s>|<unk>|[`{}\[\]>])", "", rel)
            rel = rel.strip(" ,.;")

            # 3b) keep only if it looks like a real relation
            if VALID_REL.match(rel):
                cleaned.append(rel)

        # 4) if we got at least one, keep the beam
        if cleaned:
            results.append(cleaned)

    return results

def generate_seq(
    model,
    input_text: str,
    tokenizer,
    num_beam: int = 3,
    do_sample: bool = False,
    max_new_tokens: int = 100,
):
    """
    Generate n‑best paths for a prompt:
    - Returns raw decoded strings in ["paths"] (with <PATH>, </PATH>, and <SEP> intact),
    - plus their beam scores and normalized scores.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        do_sample=do_sample,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )

    # strip off prompt tokens
    gen_seqs = out.sequences[:, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(gen_seqs, skip_special_tokens=False)
    decoded = [d.strip() for d in decoded]

    # beam scores
    if num_beam > 1:
        scores = out.sequences_scores.tolist()
        norm_scores = torch.softmax(out.sequences_scores, dim=0).tolist()
    else:
        scores = [1.0]
        norm_scores = [1.0]

    return {"paths": decoded, "scores": scores, "norm_scores": norm_scores}

def check_graph(a_entity, graph):
    flattened_graph = [item for sublist in graph for item in sublist]
    matches = [entity for entity in a_entity if entity in flattened_graph]
    return matches

def generate_subgraph(logpath, G, question, entity, id):
    out = f"{logpath}/failed_ids.json"
    sub1_nodes = set()
    try:
        sub1_nodes.add(entity)
        neighbors_1hop = list(G.neighbors(entity))
        sub1_nodes.update(neighbors_1hop)
        sub1 = G.subgraph(sub1_nodes)
        for i in neighbors_1hop:
            neighbors_2hop = list(G.neighbors(i))
            sub1_nodes.update(neighbors_2hop)
        sub2 = G.subgraph(sub1_nodes)
        return sub1, sub2
    except nx.NetworkXError:
        line = {
            "failed_id:": id,
            "question:": question,
            "entity in q_entity": entity
        }
        with open(out, "a") as f:
            f.write(json.dumps(line) + "\n")
            f.flush()
        f.close()
        print(f"Question entity '{entity}' not found. Saved question ID: {id}")
        return None, None
    
def visualize_graph(G, q_entity, a_entity, filename):
    node_colors = [
        "red" if node in q_entity else 
        "green" if node in a_entity else
        "lightblue" for node in G.nodes()
    ]
    pos = nx.spring_layout(G, k=0.5)
    plt.figure(figsize=(12, 12))
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=8
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "relation"),
        font_color="red",
    )
    plt.savefig(filename)
    plt.close()
    print(f"subraph saved in {filename}")

def gen_prediction(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
        print("Load LORA model")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN,
        )

    # input_file = os.path.join(args.data_path, args.d)
    input_file = "data/QoG.jsonl"
    print("path!!!!!",input_file)
    dataset = load_dataset(
        "json",
        data_files=input_file,
        split="train" 
    )
    print(dataset)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
 
    print("Save results to: ", output_dir)

    # # Load dataset
    # dataset = load_dataset(input_file, split=args.split)
    

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)
    # use prepare_dataset() as the local function
    def prepare_dataset(sample):
        # Prepare input prompt, get ["text"] with prompt and question
        sample["text"] = prompter.format(
            instruction=INSTRUCTION, message=sample["question"]
        )
        # Find ground-truth paths for each Q-P pair, get ["graph"] directly
        graph = utils.build_graph(sample["graph"])
        # get the shortest path with q_entity and a_entity
        paths = utils.get_truth_paths([sample["q_entity"]], sample["a_entity"], graph)
        ground_paths = set()
        for path in paths:
            ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
        # get "ground_paths" z
        sample["ground_paths"] = list(ground_paths)
        return sample
    # call prepare_dataset directly without "sample" parameter:iterates through each sample in dataset
    dataset = dataset.map(
        prepare_dataset,
        num_proc=N_CPUS,
    )
    print("after:", dataset)    

    # # get the 2-hop subgraph of specific question entity
    # dataset_name = args.d[4:]

    # input_id = "WebQTest-6"
    # # print(f"@get the subgraph of {input_id} in dataset {dataset_name}")
    # mypath = f"log/{dataset_name}/size3.json"
    # out = f"log/{dataset_name}"
    # with open(mypath, "a") as f:
    #     for row in dataset:
    #         q_entity = row["q_entity"]
    #         question = row["question"]
    #         a_entity = row["a_entity"]
    #         G = utils.build_graph(row["graph"]) # nx.Graph()
    #         q_id = row["id"]
    #         # if q_id == input_id:
    #         if nx.is_connected(G):
    #             diameter = nx.diameter(G)
    #         else:
    #             diameter = 0
            
    #         # print("@diameter of searched KG", diameter)
    #         for entity in q_entity:
    #             sub1, sub2 = generate_subgraph(logpath=out, G = G, question=question, entity = entity, id=q_id)

    #         if sub2 != None: 
    #             if nx.is_connected(sub2):
    #                 diameter2 = nx.diameter(sub2)
    #             else:
    #                 diameter2 = 0
    #             # print("@diameter of 2-hop subgraph", diameter1)
    #             res = {
    #                 "id":row["id"],
    #                 "kg": [G.number_of_nodes(), G.number_of_edges(), diameter], 
    #                 "1-hop": [sub1.number_of_nodes(), sub1.number_of_edges()],
    #                 "2-hop": [sub2.number_of_nodes(), sub2.number_of_edges(), diameter2],
    #                 "compare":  G.number_of_nodes() == sub2.number_of_nodes() and G.number_of_edges() == sub2.number_of_edges()
    #                 }
                
    #             f.write(json.dumps(res) + "\n")
    #             f.flush()
    #         else:
    #             pass
    # f.close()

    # print('size comapre complete saved in', mypath)
        # # compare a_entity and graph
    # answers = []
    # unsolved_list = []
    # for row in dataset:
    #     a_entity = row['a_entity']
    #     graph = row["graph"]
    #     matches = check_graph(a_entity, graph)
    #     answers.append([row['id'], row['question'], a_entity, matches])
    #     # if len(matches)!= len(a_entity):
    #     if len(matches) == 0:
    #         unsolved_list.append(row['id'])
    # count_unsolve = len(unsolved_list)

    
    # summary = {
    #     "Unsolved": count_unsolve,
    #     "ID":unsolved_list,
    #     "details": answers
    # }
    # output = os.path.join("log/cwq","upper_count.json")
    # with open(output, "w") as f:
    #     json.dump(summary, f, indent=4)
    
    # print('Unsolved Count',count_unsolve)

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(
        output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl"
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    
    for data in tqdm(dataset):
        question = data["question"]
        input_text = data["text"]
        qid = data["id"]
        if qid in processed_results:
            continue
        raw_output = generate_seq(
            model,
            input_text,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )
        rel_paths = parse_prediction(raw_output["paths"])
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Prediction: ", rel_paths)
        # prediction = outputs[0]["generated_text"].strip()
        data = {
            "id": qid,
            "question": question,
            "prediction": rel_paths,
            "ground_paths": data["ground_paths"],
            "input": input_text,
            "raw_output": raw_output,
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
    f.close()

    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="lianglz"
    )
    parser.add_argument("--d", "-d", type=str, default="QoG")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="Llama-2-7b-hf", # Llama-3.1-8B-Instruct
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default="/data/liang/workspace/KG/train/wiki", #meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/llama2.txt"
    )
    parser.add_argument(
        "--rel_dict",
        nargs="+",
        default=["datasets/KG/fbnet/relations.dict"],
        help="relation dictionary",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--lora", action="store_true", help="load lora weights")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=1) # default=1
    parser.add_argument("--do_sample", action="store_true", help="do sampling")

    args = parser.parse_args()

    gen_path = gen_prediction(args)
