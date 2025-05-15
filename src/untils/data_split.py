from datasets import Dataset, DatasetDict
import json
import os
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

data_path = "output/data.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]


full_dataset = Dataset.from_list(data)

# splits
from sklearn.model_selection import train_test_split
train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, dev = train_test_split(train_val, test_size=0.1111, random_state=42)  # ≈10% dev

train_dataset = Dataset.from_list(train)
dev_dataset = Dataset.from_list(dev)
test_dataset = Dataset.from_list(test)

dataset_dict = DatasetDict({
    "full": full_dataset,
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_dataset
})

for split in dataset_dict:
    print(f"{split}: {len(dataset_dict[split])} examples")

# Push to Hugging Face Hub
dataset_dict.push_to_hub("lianglz/KGQAGen-10k", token=hf_token)
