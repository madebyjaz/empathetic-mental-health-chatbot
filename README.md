# Empathetic Chatbot (Pending Title)

- This NLP powered, Django with Streamlit chatbot detects a user emotions, intent, empathy needs, and mental-health risk signals, then responds with supportive dialouge - and shows emotion/risk trends over time in a dashboard.

### What is the goal of this project??

- The goal is to **augment** human support, not replace it entirely. This allows an option for users to vent recieve some type of counsel when no other option is availbe to them. 

---
## Background
Empathetic dialogue is still hard for standard chatbots — they’re good at *what to say*, weaker at *how to say it*. This project draws on three key papers:

- Rashkin et al. 2019 — **EMPATHETICDIALOGUES** benchmark for open-domain empathetic responses.  
- Tu et al. 2022 — **MISC**, which mixes strategies + commonsense (COMET) for emotional support.  
- Zhou et al. 2018 — **Emotional Chatting Machine (ECM)**, which adds internal/external emotion memory for more human-like responses.  
These papers show that explicitly modeling **emotion + strategy** improves empathy and user satisfaction.

---
## Features 
- **Multi-class emotion detection** using `google/electra-base-goemotions`.
- **Dashboard** to visualize emotion/sentiment/risk trends per user.


## Architecture 
#### NLP Classifier Pipeline
Trained and fine-tuned on iTiger GPU cluster. Models are later downloaded and loaded locally in Streamlit.

| Category | Models & Datasets |
|-----------|--------|
| **Emotion & Sentiment Classifiers** | `google/electra-base-goemotions`, LIWC (licensed), HF Sentiment Pipeline, `roberta-large` (SARC) |
| **Intent & Topic Classifiers** | `BART-Large-MNLI` or `mDeBERTa-v3-XNLI`, fine-tuned `roberta-base`/`deberta-v3-base` |
| **Empathy & Dialogue Strategy Classifiers** | `roberta-base` token |
| **Risk & Safety Classifiers** | `deberta-v3-large` + temporal pooling, Connotation Frames, safety guard |
| **Context & Temporal Memory Classifiers** | `longformer-base` / RAG-style encoder, 1D-CNN for emotion trend deterioration |

<br/>

Here are more details for each category:
<br/>

- #### Emotion & Sentiment Classifiers
    - `google/electra-base-goemotions` -> Multi-class emotion classification w/ 27 labels
    - `LIWC` -> licensed pscholinguistic lexicon to provide an emotional effect and tone for users 
    - Generic Sentiment Pipeline via Hugging Face -> Polarity scoring and baseline mood tracking 
    - `roBERTa-large ` model fined tuned using the `SARC (Reddit)` dataset -> provides detection for sarcasm and irony
  - #### Intent and Topic Classifiers 
    - `BART-large-MNLI` or `mDeBERTa-v3-XNLI` -> Zero-shot classification for user user intent 
    - `roBERTa-base` fine tuned <b><u>or</u></b> `DeBERTa-v3-base` -> Mental health topic classification with user-level aggregation (mean/EMA)
  - #### Empathy and Dialouge Strategy 
    - `roBERTa-base` token classifier -> A multi-task model for emotion and strategy recognition (e.g. reflection, validation, reassurance, etc.)
  - #### Risk and Safety 
    - `roBERTa-large` <b><u>or</u></b> `deBERTa-v3-large` w/ temporal pooling -> crisis and self harm detection
    - Connotation frames for tone monitoring and blame/praise detection (because trust me we all need it the way some of yall be siking yall self out is crazy🤣)
    - Safety Production Layer -> consisting of two stages:
      - a fast forward keyword/regex guard
      - a neural crisis detector which triggers an escalation policy (e.g. giving resources, halting the text generation) when it deems an appropiate time.

#### Streamlit Web Application

- Simple **chat interface** (`st.chat_message()`).
- **Session state** maintains conversation context and emotion history.
- **Dashboard page** (using Streamlit multipage) visualizes weekly emotion averages, sentiment polarity, and risk levels via line charts.
- All inference runs locally or through a lightweight backend service.
---

## 🖥️ Example Project Structure

```text
empathetic-chatbot/
├── app.py                   # Streamlit main chat UI
├── pages/
│   └── 01_Dashboard.py      # emotion/risk visualization
├── inference/
│   ├── inference_registry.py
│   └── utils.py
├── models/
│   ├── empathy-roberta/
│   ├── risk-deberta/
│   └── emotion-goemotions/
├── requirements.txt
└── README.md
```

## Installing & Setup
1. Clone
```text
git clone https://github.com/madebyjaz
cd 
```

2. Environment
```text
python -m chatbot_env .venv
source chatbot_env/bin/activate   # Windows: chatbot_env\Scripts\activate
pip install -r requirements.txt
```

3. Requirements
```text
streamlit
transformers
torch
pandas
seaborn
numpy 
scikit-learn
sentencepiece
```

## Evaluation

## Ethics

## Project Timeline
- Week 1 -> Finalize architecture, dataset prep, baseline emotion classifier.
- Week 2 -> Integrate sentiment, sarcasm, and intent models.
- Week 3 -> Add empathy and safety modules.
- Week 4 -> Backend + temporal tracking.
- Week 5 -> Build Streamlit dashboard, test.
- Week 6 -> Final evaluation and report submission.

## References
- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2019). <em>  Towards empathetic open-domain conversation models: A new benchmark and dataset.</em>  ACL.
- Tu, Y., Meng, Z., Huang, M., & Zhu, X. (2022). <em> MISC: A mixed strategy-aware model integrating COMET for emotional support conversation. </em> ACL Findings.
- Zhou, H., Huang, M., Zhang, T., Zhu, X., & Liu, B. (2018). <em> Emotional chatting machine: Emotional conversation generation with internal and external memory. </em> AAAI.

## Disclaimer
- This is a research prototype intended for educational use only. It should not be used as a diagnostic or therapeutic tool.
<br/>

<b>If you or someone you know is in a crisis, please contact <em><u>988 (United States)</u></em> or your local emergency services. </b>