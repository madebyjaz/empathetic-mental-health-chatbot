#  🌟💝 Empathetic Mental Health Support Chatbot  
> _"Enhancing emotional well-being through empathetic AI conversations."_


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗_Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)


## ✨ Overview  
The **Empathetic Mental Health Support Chatbot** leverages **Natural Language Processing** and **Affective Computing** to provide emotionally intelligent, context-aware mental-health assistance.  

It recognizes user emotions, detects risk, understands intent, and responds with empathy — all while maintaining safety via a multi-stage crisis-detection pipeline.

🩵 The chatbot is **not a replacement for therapy**, but a digital companion that enhances accessibility and supports emotional awareness.
The goal is to **augment** human support for indivuals financially incapable of getting support or those who need immediate support. 

---

## ⏮️ Background
Empathetic dialogue is still hard for standard chatbots — they’re good at *what to say*, weaker at *how to say it*. This project draws on three key papers:

- Rashkin et al. 2019 — **EMPATHETICDIALOGUES** benchmark for open-domain empathetic responses.  
- Tu et al. 2022 — **MISC**, which mixes strategies + commonsense (COMET) for emotional support.  
- Zhou et al. 2018 — **Emotional Chatting Machine (ECM)**, which adds internal/external emotion memory for more human-like responses.  

These papers show that explicitly modeling **emotion + strategy** improves empathy and user satisfaction.

---

## 🧩 Features  
- [ ] 🗣 **Emotion & Sentiment Detection** (27 emotion labels using GoEmotions)  
- [ ] 💬 **Empathy Modeling** (reflection, reassurance, validation)  
- [ ] 🧭 **Intent & Topic Classification** (zero-shot + fine-tuned models)  
- [ ] 🚨 **Risk & Crisis Detection** (multi-stage neural + keyword guard)  
- [ ] 📈 **Dashboard Visualization** (emotion/sentiment/risk trends)  

---

## ⚙️ System Architecture  

### 🧠 NLP Classifier Pipeline  
| Task | Models / Libraries |
|------|--------------------|
| Emotion & Sentiment | `google/electra-base-goemotions`, LIWC, Hugging Face Sentiment |
| Sarcasm & Irony | `roberta-large-SARC` |
| Intent & Topic | `bart-large-mnli`, `mDeBERTa-v3-xnli`, fine-tuned `roberta-base` |
| Empathy & Dialogue | `roberta-base` token classifier (multi-task) |
| Risk & Safety | `roberta-large`, `deberta-v3-large`, Connotation Frames |

<br/>

Here are more details for each of the Classified Tasks:
<br/>

#### Emotion & Sentiment Classifiers
  - `google/electra-base-goemotions` -> Multi-class emotion classification w/ 27 labels
- `LIWC` -> licensed pscholinguistic lexicon to provide an emotional effect and tone for users 
-  Generic Sentiment Pipeline via Hugging Face -> Polarity scoring and baseline mood tracking 
- `roBERTa-large ` model fined tuned using the `SARC (Reddit)` dataset -> provides detection for sarcasm and irony

#### Intent and Topic Classifiers 
- `BART-large-MNLI` or `mDeBERTa-v3-XNLI` -> Zero-shot classification for user user intent 
- `roBERTa-base` fine tuned <b><u>or</u></b> `DeBERTa-v3-base` -> Mental health topic classification with user-level aggregation (mean/EMA)


#### Empathy and Dialouge Strategy 
- `roBERTa-base` token classifier -> A multi-task model for emotion and strategy recognition (e.g. reflection, validation, reassurance, etc.)

#### Risk and Safety 
- `roBERTa-large` <b><u>or</u></b> `deBERTa-v3-large` w/ temporal pooling -> crisis and self harm detection
- Connotation frames for tone monitoring and blame/praise detection (because trust me we all need it the way some of yall be siking yall self out is crazy🤣)
- Safety Production Layer -> consisting of two stages:
  - a fast forward keyword/regex guard
  - a neural crisis detector which triggers an escalation policy (e.g. giving resources, halting the text generation) when it deems an appropiate time.


### 🗂 Dialogue Management Module  
Rule-driven logic fused with classifier outputs to maintain:  
- Contextual empathy  
- Emotional coherence across turns  
- Escalation control in high-risk scenarios  

### 🌐 Streamlit Web Application  
- User-friendly chatbot interface  
- Real-time emotion & sentiment graphs  
- Secure data storage and admin dashboard  

---

## 🛠 Installating & Setting Up the Chatbot

### Prerequisites For This Project
- Python 3.10 or higher  
- pip / conda  
- Node.js (optional for frontend)  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/madebyjaz/empathetic-chatbot.git
cd empathetic-chatbot
```
### 2️⃣ Create the Virtual Environment For the Chatbot
``` bash
python -m env_name .venv
source env_name/bin/activate      # On Windows: env_name\Scripts\activate
```

### 3️⃣ Install the Necessary Dependencies

### 4️⃣ Run the Migration
``` bash
python file_name.py migrate
```

### 5️⃣ Start the Server
``` bash
python file_name.py runserver
```
Then you will be able to access the project via your web browser. You should see a message that the server is running locally on your device. Something like this:

``` 
Access the web app at: http://000.0.0.0:3000/
```
## 💻 Project Structure (Ongoing: To Be Modified)
```
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

## 🧠 Tech Stack  

| **Category** | **Tools / Frameworks** |
|---------------|------------------------|
| **Programming Language** | Python 3.10+ |
| **Web Framework / UI** | Streamlit |
| **NLP / ML Models** | Hugging Face Transformers, LIWC, GoEmotions, RoBERTa, DeBERTa |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Streamlit Charts, Seaborn |
| **Task Queue / Async** | Celery (optional) |
| **Database / Storage** | PostgreSQL / SQLite / CSV (depending on setup) |
| **Frontend Integration** | Streamlit Components, HTML/CSS (optional custom styling) |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git & GitHub |
| **Environment Management** | Virtualenv / venv / Conda |



## ⏳ Project Timeline
- [ ] Week 1 -> Finalize architecture, dataset prep, baseline emotion classifier.
- [ ] Week 2 -> Integrate sentiment, sarcasm, and intent models.
- [ ] Week 3 -> Add empathy and safety modules.
- [ ] Week 4 -> Backend + temporal tracking.
- [ ] Week 5 -> Build Streamlit dashboard, test.
- [ ] Week 6 -> Final evaluation and report submission.

## 🚀 Future Features
- [ ] 🌎 Multilingual emotion recognition
- [ ] 🎙 Speech-based emotion analysis
- [ ] 🧩 Enhanced risk escalation framework
- [ ] 🧘 Ethical AI integration & human-in-the-loop oversight

## 📚 References 
- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2019). <em>  Towards empathetic open-domain conversation models: A new benchmark and dataset.</em>  ACL.
- Tu, Y., Meng, Z., Huang, M., & Zhu, X. (2022). <em> MISC: A mixed strategy-aware model integrating COMET for emotional support conversation. </em> ACL Findings.
- Zhou, H., Huang, M., Zhang, T., Zhu, X., & Liu, B. (2018). <em> Emotional chatting machine: Emotional conversation generation with internal and external memory. </em> AAAI.

## ‼️ Disclaimer ‼️
- This is a research prototype intended for educational use only. It should not be used as a diagnostic or therapeutic tool.
<br/>

<b>If you or someone you know is in a crisis, please contact <em><u>988 (United States)</u></em> or your local emergency services. </b>