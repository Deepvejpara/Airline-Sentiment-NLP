# ✈️ Airline Sentiment Analysis using NLP & Machine Learning

An end-to-end Natural Language Processing (NLP) project focused on analyzing airline-related tweets, understanding customer opinions, and classifying feedback into sentiment categories using Machine Learning techniques.

## 📌 Project Overview

Airlines receive thousands of customer opinions and complaints through social media. Analyzing this feedback manually can be time-consuming.

This project uses NLP and Machine Learning to process airline-related tweets, extract meaningful information, explore customer sentiment, and build sentiment classification models.

The primary objective is to identify whether customer feedback expresses:

- 😊 **Positive** – Satisfied customers and positive experiences.
- 😐 **Neutral** – Informational or neutral feedback.
- 😠 **Negative** – Complaints, dissatisfaction, and negative experiences.

## 🌐 Streamlit Application

An interactive Streamlit application is planned to demonstrate the sentiment analysis pipeline through a user-friendly interface.

**Planned Features:**
- Enter custom airline-related text.
- Generate sentiment predictions.
- Display predicted sentiment categories.
- Explore model predictions interactively.
- Visualize relevant sentiment insights.

**Live Application:** Coming Soon

<!https://airline-sentiment-nlp-zacxzbmlw6yttonezqz3fe.streamlit.app/>

## 🎥 Video Walkthrough

A video walkthrough will demonstrate the complete project, including data preprocessing, NLP techniques, feature extraction, model development, evaluation, and application functionality.

**Project Demonstration:** Coming Soon

<!https://drive.google.com/drive/folders/1GEcYqlI21NEHEK0yYanRAjpjnvwyY_UI?usp=drive_link>

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess raw text data.
- Apply NLP techniques for text normalization.
- Analyze word frequencies and common expressions.
- Implement Bag of Words and TF-IDF vectorization.
- Explore N-gram features and Word2Vec embeddings.
- Train and evaluate Machine Learning classifiers.
- Analyze model performance and classification errors.
- Build an interactive sentiment analysis application using Streamlit.

## 📊 Dataset

**Dataset:** Twitter US Airline Sentiment

The dataset contains airline-related tweets labeled according to customer sentiment.

The analysis focuses on understanding customer feedback and identifying patterns across different sentiment categories and airlines.

## ⚙️ Project Workflow

### 1. Data Collection & Exploration
- Load the airline sentiment dataset.
- Inspect dataset structure and dimensions.
- Examine missing values and duplicate records.
- Explore sentiment distribution.
- Analyze airline-wise customer feedback.

### 2. Text Preprocessing
- Text cleaning using Regular Expressions.
- Lowercase normalization.
- Tokenization.
- Stopword removal.
- Stemming and Lemmatization.
- Text normalization and preparation.

### 3. Exploratory Data Analysis
- Sentiment distribution analysis.
- Word frequency analysis.
- Most frequently used words.
- Airline-wise sentiment exploration.
- Visualization of common customer expressions.

### 4. Text Feature Extraction

Explore different approaches to convert textual information into numerical representations:

- Bag of Words (BoW)
- Term Frequency–Inverse Document Frequency (TF-IDF)
- N-gram representation
- Word2Vec embeddings

### 5. Machine Learning

Apply Machine Learning classification techniques to identify sentiment categories from processed text.

The workflow includes:
- Model training and comparison.
- Train-test splitting.
- Cross-validation.
- Classification reports.
- Confusion matrix analysis.
- Prediction and error analysis.

### 6. Model Evaluation

Evaluate classification performance using appropriate metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation process helps understand model strengths and areas requiring improvement.

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | NLP preprocessing |
| Scikit-learn | Feature extraction and ML |
| Regular Expressions | Text cleaning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| WordCloud | Text visualization |
| Gensim | Word embeddings |
| Streamlit | Interactive application |

## 📂 Repository Structure

```text
Airline-Sentiment-NLP/
│
├── data/
│   └── Dataset files
│
├── notebooks/
│   └── NLP analysis and model development
│
├── app/
│   └── Streamlit application (planned)
│
├── README.md
└── requirements.txt
```

*Note: This is a reference structure; adjust it to match the final repository.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Deepvejpara/Airline-Sentiment-NLP.git
```

### 2. Navigate to the Project

```bash
cd Airline-Sentiment-NLP
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Notebook

Open the Jupyter Notebook and execute the cells to explore the complete NLP workflow.

```bash
jupyter notebook
```



## 📚 Key Learning Outcomes

Through this project, I explored:

- Practical applications of Natural Language Processing.
- Text preprocessing and normalization techniques.
- Different text vectorization approaches.
- Word embeddings and semantic representations.
- Machine Learning-based text classification.
- Model evaluation and error analysis.
- Building interactive applications with Streamlit.

## 👨‍💻 Author

**Deep Vejpara**

BSc IT | Data Science & Machine Learning Enthusiast

- GitHub: [Deepvejpara](https://github.com/Deepvejpara)

---

⭐ If you find this project interesting, consider giving the repository a star!
