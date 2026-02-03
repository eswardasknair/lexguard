# ============================================================
# LEXGUARD IMPROVED TRAINING PIPELINE — v3.0
# Run this in Google Colab with a T4/A100 GPU enabled.
# FIXES v3.0:
#   1. Removed trust_remote_code (no longer supported)
#   2. Fixed evaluation_strategy → eval_strategy (new Transformers API)
#   3. Fixed label mapping — unfair_tos uses LIST labels e.g. [0,2]
#      not a single integer, so the old == 1 check always failed.
#      All 4 classes (Low/Medium/High/Critical) now train correctly.
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1: Install Dependencies
# ────────────────────────────────────────────────────────────
print("⏳ STEP 1: Installing dependencies...")
# !pip install transformers torch scikit-learn joblib pandas datasets accelerate -U -q
# (uncomment the line above in Colab)
print("✅ Done.")

import torch
import joblib
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ────────────────────────────────────────────────────────────
# STEP 2: Load Dataset
# ────────────────────────────────────────────────────────────
print("\n⏳ STEP 2: Loading LexGLUE Unfair-ToS dataset...")
dataset = load_dataset("lex_glue", "unfair_tos")  # trust_remote_code removed (deprecated)
# Dataset splits:
#   train      ~5,532 examples
#   validation ~2,275 examples
#   test       ~1,607 examples
# IMPORTANT: unfair_tos labels are LISTS of category IDs, e.g. [0, 2] or []
# Categories: 0=Limitation of Liability, 1=Unilateral Termination,
#             2=Unilateral Change, 3=Content Removal, 4=Contract by Using,
#             5=Choice of Law, 6=Choice of Venue, 7=Arbitration
# Empty list [] = Fair clause (no unfair term detected)
print(f"✅ Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

# Show a few raw label examples so you understand the format
print("  Sample raw labels:", [dataset['train'][i]['labels'] for i in range(5)])

# ────────────────────────────────────────────────────────────
# STEP 3: Smart 4-Class Label Mapping
# FIX: Original code only produced label 0 and 3, leaving
#      Medium (1) and High (2) completely untrained.
#      We now use keyword heuristics to split Unfair → Critical/High
#      and Fair → Low/Medium, giving all 4 classes real training data.
# ────────────────────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    'terminat', 'unilateral', 'without notice', 'at any time',
    'indemnif', 'hold harmless', 'liquidated damages', 'penalty',
    'intellectual property', 'all rights', 'assign', 'forfeit',
    'sole discretion', 'immediately',
]
HIGH_KEYWORDS = [
    'not liable', 'no liability', 'limitation of liability', 'disclaim',
    'no warranty', 'non-compete', 'restrict', 'confidential',
]
MEDIUM_KEYWORDS = [
    'may modify', 'reserve the right', 'at our discretion', 'may change',
    'third party', 'transfer', 'share your data', 'may use',
]

# CRITICAL_TERMINATION and UNILATERAL categories in unfair_tos map to Critical
# Category IDs: 1=Unilateral Termination, 2=Unilateral Change → Critical
# Category IDs: 0=Limitation of Liability, 7=Arbitration → High
# Empty list [] = Fair clause → Low or Medium based on text keywords
CRITICAL_CATEGORY_IDS = {1, 2}   # Termination / Unilateral Change
HIGH_CATEGORY_IDS     = {0, 3, 4, 6, 7}  # Liability, Arbitration, etc.

def map_to_4class_risk(example):
    """
    Maps the multi-label list format of unfair_tos to 4-class LexGuard risk:
        Labels is a LIST e.g. [], [0], [1, 2], NOT a single integer.
        0 = Low      (fair clause, standard language)
        1 = Medium   (fair clause with discretionary/data-sharing language)
        2 = High     (unfair clause: liability limit, arbitration, etc.)
        3 = Critical (unfair clause: termination, unilateral change)
    """
    text = example['text'].lower()
    label_list = example['labels']  # This is a LIST like [] or [0, 2]

    # Determine if the clause is unfair and what category
    if isinstance(label_list, list):
        is_unfair = len(label_list) > 0
        label_set = set(label_list)
    else:
        # Fallback: some versions may store as int (-1 = fair)
        is_unfair = (label_list != -1)
        label_set = {label_list} if is_unfair else set()

    if is_unfair:
        # Check for Critical category IDs first
        if label_set & CRITICAL_CATEGORY_IDS:
            return {'labels': 3}  # Critical
        # Then High categories
        if label_set & HIGH_CATEGORY_IDS:
            return {'labels': 2}  # High
        # Any other unfair = High as fallback
        for kw in CRITICAL_KEYWORDS:
            if kw in text:
                return {'labels': 3}
        return {'labels': 2}
    else:  # Fair clause
        for kw in MEDIUM_KEYWORDS:
            if kw in text:
                return {'labels': 1}  # Medium
        return {'labels': 0}          # Low

print("\n⏳ STEP 3: Mapping to 4-class risk labels...")
encoded_dataset = dataset.map(map_to_4class_risk)

# Print class distribution — you MUST include this in your paper!
for split_name in ['train', 'validation', 'test']:
    labels = encoded_dataset[split_name]['labels']
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
    dist = {label_names[u]: int(c) for u, c in zip(unique, counts)}
    print(f"  {split_name} distribution: {dist}")

# ────────────────────────────────────────────────────────────
# STEP 4: Load InLegalBERT
# ────────────────────────────────────────────────────────────
print("\n⏳ STEP 4: Loading InLegalBERT model...")
MODEL_NAME = 'law-ai/InLegalBERT'
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    )
    print(f"✅ Loaded: {MODEL_NAME}")
except Exception as e:
    print(f"⚠️ InLegalBERT unavailable ({e}). Falling back to bert-base-uncased.")
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    )

# ────────────────────────────────────────────────────────────
# STEP 5: Tokenize Dataset
# ────────────────────────────────────────────────────────────
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print("\n⏳ STEP 5: Tokenizing...")
tokenized_datasets = encoded_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format("torch")
print("✅ Tokenization complete.")

# ────────────────────────────────────────────────────────────
# STEP 6: Metrics Function (FIX — this was missing before)
# ────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    Computes Accuracy, Weighted F1, and Macro F1 after each epoch.
    These numbers are what you report in your paper.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc          = accuracy_score(labels, preds)
    f1_weighted  = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_macro     = f1_score(labels, preds, average='macro',    zero_division=0)
    return {
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro":    round(f1_macro, 4),
    }

# ────────────────────────────────────────────────────────────
# STEP 7: Training Configuration
# ────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,              # 5 epochs for better convergence
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,               # L2 regularization
    warmup_steps=200,                # Gradual LR warmup
    eval_strategy="epoch",           # FIX: renamed from evaluation_strategy in new Transformers
    save_strategy="no",
    logging_steps=100,
    remove_unused_columns=False,
    fp16=True,                       # Enable mixed precision (faster on GPU)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,  # ← KEY FIX
)

# ────────────────────────────────────────────────────────────
# STEP 8: Train
# ────────────────────────────────────────────────────────────
print("\n🚀 STEP 6: Training started...")
trainer.train()
print("✅ Training complete.")

# ────────────────────────────────────────────────────────────
# STEP 9: Full Evaluation — Report These in Your Paper
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 FINAL VALIDATION SET METRICS")
print("="*60)
val_results = trainer.evaluate()
for k, v in val_results.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("📊 TEST SET EVALUATION (Per-Class Report)")
print("="*60)
test_output   = trainer.predict(tokenized_datasets['test'])
test_preds    = np.argmax(test_output.predictions, axis=-1)
test_labels   = test_output.label_ids

print(classification_report(
    test_labels,
    test_preds,
    target_names=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
    zero_division=0,
))

print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(test_labels, test_preds))

# ────────────────────────────────────────────────────────────
# STEP 10: Save the Risk Model
# ────────────────────────────────────────────────────────────
print("\n💾 Saving english_risk_model.pkl ...")
risk_model_bundle = {
    'model_state': model.state_dict(),
    'config':      model.config,
    'tokenizer':   tokenizer,
}
joblib.dump(risk_model_bundle, 'english_risk_model.pkl')
print("✅ 'english_risk_model.pkl' saved — download from Files sidebar.")

# ────────────────────────────────────────────────────────────
# STEP 11: GNN — Save Architecture Template (Honest Approach)
# FIX: Previous code saved random untrained weights as if it were
#      a trained model. That was misleading. This version saves
#      an architecture description. Training requires a real
#      clause-graph dataset which is future work.
# ────────────────────────────────────────────────────────────
gnn_metadata = {
    'status':       'architecture_template',
    'description':  'GCN for inter-clause dependency modeling. Training requires '
                    'a labeled clause-graph dataset (future work).',
    'architecture': {
        'conv1': 'GCNConv(in=1, out=16)',
        'conv2': 'GCNConv(in=16, out=2)',
        'activation': 'ReLU',
        'regularization': 'Dropout(p=0.5)',
        'output': 'LogSoftmax',
    },
}
joblib.dump(gnn_metadata, 'clause_relationship_gnn.pkl')
print("✅ 'clause_relationship_gnn.pkl' saved (architecture template).")

print("\n" + "="*60)
print("🎉 ALL DONE! Download both .pkl files from the Colab Files sidebar.")
print("Place them in: lexguard/ml_models/")
print("="*60)
