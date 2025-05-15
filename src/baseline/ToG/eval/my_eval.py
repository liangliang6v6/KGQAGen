import argparse, json, re, string
from datasets import load_dataset
from collections import Counter

def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def match(pred, gt):
    return normalize(gt) in normalize(pred)

def accuracy(pred, gts):
    return sum(match(pred, g) for g in gts) / len(gts)

def hit1(pred, gts):
    return int(any(match(pred, g) for g in gts))

def f1_score(preds, gts):
    if not preds:
        return 0.0, 0.0, 0.0
    ps = " ".join(preds)
    m = sum(match(ps, g) for g in gts)
    p = m / len(preds)
    r = m / len(gts)
    return (2*p*r/(p+r) if p+r else 0.0), p, r

def extract_pred(text):
    m = re.findall(r"\{([^{}]+)\}", text)
    return m[-1].strip() if m else ""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   type=str, default="lianglz/KGQAGen-10k")
    p.add_argument("--raw",       type=str, default="ToG.jsonl")
    p.add_argument("--out_prefix",type=str, default="eval_ToG2")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split="test")
    # map question → (id, ground_truth list)
    gt_map = {
        item["question"]:
          (item["id"],
           list(item["answer"].values()) if isinstance(item["answer"], dict)
           else item["answer"])
        for item in ds
    }

    detailed = f"{args.out_prefix}_detailed.jsonl"
    summary  = f"{args.out_prefix}_summary.txt"

    accs, hits, f1s, ps, rs = [], [], [], [], []
    seen_ids = set()

    with open(args.raw, "r", encoding="utf-8") as fin, open(detailed, "w", encoding="utf-8") as fout:
        for line in fin:
            j = json.loads(line)
            q = j["question"]
            pred = extract_pred(j.get("results", ""))
            if q not in gt_map:
                continue
            idx, gts = gt_map[q]
            if idx in seen_ids:
                continue
            seen_ids.add(idx)

            a = accuracy(pred, gts)
            h = hit1(pred, gts)
            f, p, r = f1_score([pred], gts)

            # **Append** to our lists so we can average later
            accs.append(a)
            hits.append(h)
            f1s.append(f)
            ps.append(p)
            rs.append(r)

            out = {
                "id": idx,
                "question": q,
                "prediction": [pred] if pred else [],
                "ground_truth": gts,
                "acc": round(a, 3),
                "hit": h,
                "f1": round(f, 3),
                "precision": round(p, 3),
                "recall": round(r, 3)
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    N = len(accs)
    if N == 0:
        print("No valid examples found; cannot compute metrics.")
    else:
        s = (
            f"Accuracy: {100*sum(accs)/N:.12f}  "
            f"Hit@1: {100*sum(hits)/N:.12f}  "
            f"F1: {100*sum(f1s)/N:.12f}  "
            f"P: {100*sum(ps)/N:.12f}  "
            f"R: {100*sum(rs)/N:.12f}"
        )
        print(s)
        with open(summary, "w", encoding="utf-8") as f:
            f.write(s + "\n")
