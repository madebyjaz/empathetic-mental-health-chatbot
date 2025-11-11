import argparse as a
from typing import Dict, Any
import numpy as num
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def parse_args():
    parser = a.ArgumentParser(description= "Train these Classifiers for EmpatheticMental Health Chat Application")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="The task to train: emotion_geoemotions, sarcasm_detection, topic_mental_health, topic_mentalhealth, empathy_strategy, risk_crisis, blame_attribution, safety_guardrails, safety_guardrails_neural, trend_deterioration_analysis",
    )
    parser.add_arguement(
        "train_file",
        type=str,
        help="Path to the training data file (CSV or JSON format)",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/", 
        help="Directory to save the models trained" 
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=8,              # could be adjusted (possibly 16) based on itiger's GPU memory
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,              # more epochs = longer training time but potentially better performance
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,           # typical lr for fine-tuning transformers
        help="Learning rate for the optimizer"
    )
    return parser.parse_args()

# Task Configs
TASK_CONFIGS = {

# 1.) Emotion & Sentiment Analysis Task


    "emotion_geoemotions": {
        "model_name": "google/electra-base-geoemotions",
        "num_labels": 27,
        "problem_type": "multilabel",
        "dataset": "geoemotions",
        "metric": "f1-micro",
        "trainable": True,
    },

    # Sarcasm / Irony Detection on SARC (Reddit) dataset
    "sarcasm_detection": {
        "model_name": "roberta-large",
        "num_labels": 2,        #sarcastic or not sarcastic
        "problem_type": "binary",
        "dataset": "sarc",
        "metric": "f1-macro",
        "trainable": True,
    },

    # Sentiment Analysis (HF Pipeline)
    "sentiment_generic": {
        "model_name": "distilbert-base-uncased",
        "num_labels": 3,        #positive, negative, neutral
        "problem_type": "multiclass",
        "dataset": "custom_sentiment",
        "metric": "f1-macro",
        "trainable": False, # This is going to be used as a pre-trained pipeline w/o further training
    },

    # LIWC Emotion and Tone Analysis (Lexicon-based), not a transformer model
    "liwc_emotion_and_tone": {
        "model_name": "LIWC/emotion-and-tone",
        "num_labels": None,
        "problem_type": "lexicon", 
        "dataset": "liwc", 
        "metric": None,
        "trainable": False,
    },


# 2.) Intent & Topic Classifiers 


    # Zero-shot intent using NLI model w/no fine-tuning
    "intent_zero_shot": {
        "model_name": "facebook/bart-large-mnli",       # or possibly "microsoft/deberta-v3-xsmall"
        "num_labels": None,
        "problem_type": "zeroshot",
        "dataset": "custom_intent_prompts",
        "metric": "f1-macro",
        "trainable": False,
    },

    # fine tuned intent classifier
    "topic_mental_health": {
        "model_name": "roberta-base", 
        "num_labels": 8,                # anxiety, depression, stress, trauma, substance use, eating disorders, bipolar disorder, schizophrenia
        "problem_type": "multilabel",
        "dataset": "mental_health_topics",
        "metric": "f1-micro",
        "trainable": True,
    },

    "topic_mentalhealth" : {
        "model_name": "microsoft/deberta-v3-base",
        "num_labels": 8,
        "problem_type": "multilabel",
        "dataset": "mental_health_topics",
        "metric": "f1-micro",
        "trainable": True,  
    },


# 3.) Empathy & Dialogue Classifier


    "empathy_strategy": {
        "model_name": "roberta-base",
        "num_labels": 8,                # reflection, question, validation, reassurance, sympathy, information, advice, etc.
        "problem_type": "multilabel",
        "dataset": "empathy_strategies",
        "metric": "f1-micro",
        "trainable": True,
    },


# 4.) Risk & Safety Classifiers

    "risk_crisis": {
        "model_name": "microsoft/deberta-v3-large",
        "num_labels": 2,                # crisis /self harm or no crisis
        "problem_type": "binary",
        "dataset": "clpsych_or_similar",
        "metric": "auroc",
        "trainable": True,
    },

    # blame / praising attribution classifier
    "blame_attribution": {
        "model_name": "roberta-base",
        "num_labels": 3,                    # blame, praising , or no blame
        "problem_type": "multiclass",
        "dataset": "blame_connotation_frames",
        "metric": "f1-macro",
        "trainable": True,
    },

    # safety guardrails using rule-based + neural keywords 
    "safety_guardrails": {
        "model_name" : "keyword_regex_rules",
        "num_labels": None,
        "problem_type": "rule_based",
        "dataset": "handcrafted",
        "metric": None,
        "trainable": False,              
    },

    # the neural crisis dector will be the risk classifier (risk_crisis)
    "safety_guardrails_neural": {
        "model_name" : "microsoft/deberta-v3-large",
        "num_labels": 2,                #safety risk or no risk
        "problem_type": "binary",
        "dataset": "clpsych_or_similar",
        "metric": "auroc",
        "trainable": False,             # trained by the first classifier (risk_crisis)
    },

# 5.) Context & Temporal Memory
    "context" : {                   #  (may not train in this file)
        "model_name": "allenai/longformer-base-4096",
        "num_labels": None, 
        "problem_type": "encoder_only",
        "dataset": None,
        "metric": None,
        "trainable": False,
    },

    "trend_deterioration_analysis": { # 1D-CNN over time to detect deterioration trends in emotion scores (no text classification)
        "model_name": "1d_cnn_timeseries",
        "num_labels": 2,                #stable or deteriorating
        "problem_type": "binary",
        "dataset": "emotion_timeseries", 
        "metric": "f1-macro", 
        "trainable": True,          # this will require a different approach (another training loop)
    },
}

def computing_metrics(eval_pred):
    predictions, labels = eval_pred
    predicts = predictions.argmax(-1)

    ac = accuracy_score(labels, predicts)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicts, average='weighted')

    metrics = {"accuracy": ac, "precision": precision, "recall": recall, "f1": f1}
    try: 
        metrics["auroc"] = roc_auc_score(labels, predictions[:, 1]) 

    except Exception as e:
        pass

    return metrics

def main():
    args = parse_args()
    if args.task not in TASK_CONFIGS:
        raise ValueError(f"Task {args.task} not is not recognized within the dictionary of Task Configs. Here are the available tasks: {list(TASK_CONFIGS.keys())}")
    
    task_in_config = TASK_CONFIGS[args.task]
    if not task_in_config["trainable"]:
        print(f"The task the user has selected ({args.task}) is not a trainable model. Please educate the user on the available options & edit the task selection. Now exiting the training process...")
        return

    print(f"👾Training Task {args.task}")
    print(f"📈Using model {task_in_config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(task_in_config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        task_in_config["model_name"],
        num_labels= task_in_config["num_labels"],
        problem_type= "multi_label_classification" if task_in_config["problem_type"] == "multilabel" else None,
    )

    dataset = load_dataset(task_in_config["dataset"], data_files={"train": args.train_file, "validation": args.val_file})

    def preprocess(batch):
        return tokenizer(batch["text"], truncation = True, padding = "max_length", max_length = 512)
    encoded_dataset = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir = args.output_dir + f"/{args.task}",
        evaluation_strategy = "epoch",
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        weight_decay = 0.01,
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "f1" if task_in_config["metric"] != "auroc" else "auroc",
        report_to = "none",
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = encoded_dataset["train"],
        eval_dataset = encoded_dataset["validation"],
        compute_metrics = computing_metrics,
        tokenizer = tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir + f"/{args.task}")
    tokenizer.save_pretrained(args.output_dir + f"/{args.task}")
    print(f"✅ Training for task {args.task} completed and model saved to {args.output_dir}/{args.task}")

if __name__ == "__main__":
    main()
