import argparse
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