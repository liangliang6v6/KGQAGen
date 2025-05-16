# Running Baselines

This section provides instructions for running four baseline methods used in our experiments: ToG, PoG, GCR, and RoG. We provide adapted versions of each method to run on our dataset, along with options for using either an online or local Wikidata endpoint.

---

## 1. Running ToG

Our implementation is adapted from the [original ToG repository](https://github.com/GasolSun36/ToG) with minor modifications for compatibility.

### Option 1: Use the Online Wikidata API

Run the following script:

```bash
bash src/baseline/ToG/ToG/run.sh
```

### Option 2: Use a Local Wikidata Query Endpoint

(Instructions for setting up a local endpoint should be added here.)

---

## 2. Running PoG

Our implementation is adapted from the [original PoG repository](https://github.com/liyichen-cly/PoG/tree/main).

### Option 1: Use the Online Wikidata API

Run the following script:

```bash
bash src/baseline/PoG/PoG/run.sh
```

### Option 2: Use a Local Wikidata Query Endpoint

Note: The original PoG implementation does not include code for setting up a local endpoint. However, since PoG is adapted from [ToG](https://github.com/GasolSun36/ToG), you can follow the same procedure used for ToG.

---

## 3. Running GCR

Our GCR implementation is based on the [original GCR repository](https://github.com/RManLuo/graph-constrained-reasoning). Full usage guidelines are available in:

```bash
src/baseline/graph-constrained-reasoning/README.md
```

### Step 0: Install Requirements

1. Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Create and activate a Conda environment:

```bash
conda create -n GCR python=3.12
conda activate GCR
poetry install
```

3. Install Flash-Attention for optimized decoding:

```bash
pip install flash-attn --no-build-isolation
```

### Step 1: Build Graph Index

```bash
bash src/baseline/graph-constrained-reasoning/scripts/build_graph_index.sh
```

### Step 2: Train the KG-Specialized LLM (LLaMA-3.1-8B default)

```bash
bash src/baseline/graph-constrained-reasoning/scripts/train_kg_specialized_llm.sh
```

### Step 3: Inference

* Graph-constrained decoding:

```bash
bash src/baseline/graph-constrained-reasoning/scripts/graph_constrained_decoding.sh
```

* Graph inductive reasoning:

```bash
bash src/baseline/graph-constrained-reasoning/scripts/graph_inductive_reasoning.sh
```

---

## 4. Running RoG

Our implementation is based on the [original RoG repository](https://github.com/RManLuo/reasoning-on-graphs).

### Step 0: Install Requirements

```bash
cd src/baseline/reasoning-on-graphs
pip install -r requirements.txt
```

### Step 1: Preprocess and Train

We use the proof steps in our dataset as subgraphs.

<details>
<summary>Data Preparation</summary>

1. Build question-to-relation path pairs:

```bash
python src/align_kg/build_align_qa_dataset.py -d lianglz/KGQAGen-10k --split train
```

2. Build joint-training datasets:

```bash
python src/joint_training/preprocess_align.py
python src/joint_training/preprocess_qa.py
```

3. Generate interpretable examples:

```bash
python src/joint_training/generate_explanation_results.py
```

</details>

### Step 2: Inference (12GB+ GPU required)

#### 2.1 Planning: Generate Relation Paths

```bash
bash scripts/planning.sh
```

Or manually:

```bash
python src/qa_prediction/gen_rule_path.py \
    --split test \
    --n_beam 3
```

Output: results/gen\_rule\_path/{dataset}/{model\_name}/{split}

#### 2.2 Reasoning: Generate Answers with RoG

```bash
bash scripts/rog-reasoning.sh
```

Or manually:

```bash
python src/qa_prediction/predict_answer.py \
    --prompt_path prompts/llama2_predict.txt \
    --add_rul \
    --rule_path {rule_path}
```

Output: results/KGQA/{dataset}/{model\_name}/{split}
