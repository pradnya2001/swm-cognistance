# 🧠 Stance Detection on Social Media

A machine learning and NLP project for detecting **stance (FAVOR, AGAINST, NONE)** in social media text using classical ML models, transformer-based models, and large language models (LLMs).

---

## 📌 Overview

This project focuses on identifying the **stance of a tweet toward a specific target topic**, going beyond traditional sentiment analysis.

Unlike sentiment analysis, stance detection determines whether a user:

* Supports a topic (**FAVOR**)
* Opposes it (**AGAINST**)
* Is neutral or unclear (**NONE**)

We implement and compare:

* Classical Machine Learning models
* Transformer-based models (BERT variants)
* Large Language Models (LLMs) with reasoning

---

## 🚀 Features

* ✅ Multi-model stance detection pipeline
* ✅ Support for classical ML, transformers, and LLMs
* ✅ Chain-of-Stance (CoS) reasoning for LLMs
* ✅ Interactive **Gradio web interface**
* ✅ Real-time prediction with confidence scores

---

## 🏗️ System Architecture

The system follows this pipeline:

1. **Input**: Tweet + Target topic
2. **Preprocessing**: Cleaning, tokenization, normalization
3. **Model Selection**:

   * Classical ML (Logistic Regression, Random Forest, KNN)
   * Transformers (BERT, BERTweet, TwHIN-BERT)
   * LLMs (Phi-3, Mistral, Llama)
4. **Output**:

   * Predicted stance (FAVOR / AGAINST / NONE)
   * Confidence score

---

## 📊 Dataset

* **SemEval-2016 Task 6**
* ~4,000 annotated tweets
* Targets include:

  * Atheism
  * Climate Change
  * Feminist Movement
  * Hillary Clinton
  * Legalization of Abortion

⚠️ The dataset is **imbalanced**, with ~50% labeled as AGAINST.

---

## 🤖 Models Used

### 🔹 Classical ML

* Logistic Regression
* Random Forest
* K-Nearest Neighbors

### 🔹 Transformer Models

* BERT-base
* BERTweet (trained on 850M tweets)
* TwHIN-BERT

### 🔹 Large Language Models

* Phi-3 Mini (3.8B)
* Mistral 7B ⭐ (Best performing)
* Llama 3.1

---

## 📈 Results

| Category     | Best Model          | Macro-F1  | Accuracy |
| ------------ | ------------------- | --------- | -------- |
| Classical ML | Logistic Regression | 0.53      | 58.6%    |
| Transformer  | BERTweet            | 0.65      | 70%      |
| LLM          | Mistral 7B          | **0.775** | **80%**  |

👉 LLMs with Chain-of-Stance reasoning significantly outperform other approaches.

---

## 🧪 Key Insights

* Stance ≠ Sentiment (positive text can still be AGAINST)
* Domain-specific models (like BERTweet) perform better on social media
* LLMs provide the best performance with reasoning
* The **NONE class is hardest to predict**
* Class imbalance affects model behavior

---

## 💻 Web Interface

Built using **Gradio**, the UI allows users to:

* Input custom tweets
* Select target topic
* Choose model
* View:

  * Predicted stance
  * Confidence score
  * Probability distribution

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/stance-detection.git
cd stance-detection
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python app.py
```

Then open:

```
http://localhost:7860
```

---

## 📁 Project Structure

```
├── data/
├── models/
├── notebooks/
├── app.py
├── utils/
├── requirements.txt
└── README.md
```

---

## 👥 Team

* Pradnya Sanjeev Nidagundi
* Abishek Prakash
* Anushka Jha
* Archit Bubber
* Karthik Ponugoti
* Nirmalraju Kangeyan

---

## 🔮 Future Work

* Cross-domain generalization
* Knowledge-enhanced stance detection
* Multi-task learning (stance + sentiment)
* Explainable AI for predictions
* Real-time large-scale deployment

---

## 📚 References

* SemEval-2016 Task 6 Dataset
* BERT, BERTweet, TwHIN-BERT
* Chain-of-Stance (CoS) prompting

---

## ⭐ Acknowledgments

Arizona State University – CSE 573: Semantic Web Mining


