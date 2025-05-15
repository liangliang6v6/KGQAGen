import json
import sys

def compute_hit(preds, label):
    hits = 0
    for l in label:
        if l == preds:
            hits = 1
            break
    return hits


def compute_metrics(preds, labels, rel_dict):
    """
    Compute metrics

    Args:
        preds (list[list[str]]): list of rules
        labels (list[list[str]]): list of rules

    Returns:
        _type_: _description_
    """
    total_hall = 0
    total_rel = 0
    # Hallucination
    for rule in preds:
        for rel in rule:
            if rel not in rel_dict:
                total_hall += 1
        total_rel += len(rule)

    # Conver list to str for quick comparison
    preds = ["<SEP>".join(p) for p in preds]
    labels = ["<SEP>".join(l) for l in labels]
    hit = 0.0
    hits = 0.0
    for rule in labels:
        if rule in preds:
            hits += 1
    if hits > 0:
        hit = 1.0
    precission = hits / len(preds)
    recall = hits / len(labels)
    if (precission + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precission * recall / (precission + recall)
    return total_hall, total_rel, hit, precission, recall, f1


def eval_generation(result_path, rel_dict_path, debug=False):
    rel_dict = set()
    for rel_file in rel_dict_path:
        with open(rel_file, "r") as f:
            for line in f:
                _, r = line.strip().split("\t")
                rel_dict.add(r)
    
    
    hit_list = []
    precission_list = []
    recall_list = []
    f1_list = []
    total_predicates = 0.0
    total_hall = 0.0
    match_results_path = result_path.replace("predictions", "match_results")
    with open(result_path, "r") as f, open(match_results_path, "w") as f_out:
        for line in f:
            data = json.loads(line)
            predicates_list = data["prediction"]
            label = data["ground_paths"]
            # Skip empty questions
            if len(label) == 0:
                continue
            n_hall, n_rel, hit, precission, recall, f1 = compute_metrics(
                predicates_list, label, rel_dict
            )
            total_predicates += n_rel
            total_hall += n_hall
            hit_list.append(hit)
            precission_list.append(precission)
            recall_list.append(recall)
            f1_list.append(f1)
            if debug:
                print("Question: ", data["question"])
                print("Prediction: ", predicates_list)
                print("Label: ", label)
                print(
                    f"Hit: {hit}, Precission: {precission}, Recall: {recall}, F1: {f1}"
                )
            f_out.write(
                json.dumps(
                    {
                        "question": data["question"],
                        "prediction": predicates_list,
                        "label": label,
                        "metrics": {
                            "hit": hit,
                            "precission": precission,
                            "recall": recall,
                            "f1": f1,
                        },
                    }
                )
                + "\n"
            )
    result = "Hit: {:.4f}, Precission: {:.4f}, Recall: {:.4f}, F1: {:.4f}, #Hall: {}, #Total: {}, Hall ratio: {:.4f}".format(
        sum(hit_list) / len(hit_list),
        sum(precission_list) / len(precission_list),
        sum(recall_list) / len(recall_list),
        sum(f1_list) / len(f1_list),
        total_hall,
        total_predicates,
        total_hall / total_predicates,
    )
    print(result)
    eval_results_path = result_path.replace("predictions", "eval_result").replace(".jsonl", ".txt")
    with open(
        eval_results_path,
        "w",
    ) as f:
        f.write(result)
    return result
