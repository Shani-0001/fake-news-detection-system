# 📰 Fake News Detection System

### Machine Learning | Natural Language Processing | Python | Streamlit

A Machine Learning-based Fake News Detection System that analyzes news articles and classifies them as **Fake News** or **Genuine News** using Natural Language Processing (NLP) and Machine Learning techniques.

The application provides an interactive web interface where users can enter a news article and receive a prediction with a model confidence score.

---

## 📌 Project Overview

The spread of misinformation and fake news on digital platforms has become a major challenge. This project aims to develop a machine learning application that identifies patterns associated with fake and genuine news articles.

The system uses text preprocessing, TF-IDF vectorization and Logistic Regression to classify news content based on patterns learned from a labeled dataset.

**Important:** This system identifies learned text patterns. It does not independently verify the factual accuracy of news articles.

---

## 🎯 Project Objectives

- Detect potentially fake news articles using Machine Learning.
- Apply NLP techniques to clean and process news text.
- Train a classification model using labeled news datasets.
- Provide an interactive web application using Streamlit.
- Display predictions and model confidence scores.
- Demonstrate an end-to-end Machine Learning workflow.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| NLP | Text processing |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification algorithm |
| Joblib | Model serialization |
| Streamlit | Web application |

---

## ⚙️ Project Architecture

1. **Dataset Collection:** Load labeled fake and genuine news datasets.
2. **Data Preprocessing:** Clean news text and remove unnecessary characters.
3. **Feature Extraction:** Convert text into numerical features using TF-IDF.
4. **Model Training:** Train the Logistic Regression classifier.
5. **Model Evaluation:** Evaluate the model using accuracy, precision, recall and F1-score.
6. **Model Saving:** Save the trained model using Joblib.
7. **Frontend Development:** Build an interactive Streamlit application.
8. **Prediction:** Analyze user-provided news articles and display results.

---

## 📊 Model Performance

The trained model achieved the following results on the project's original test split:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.63% |
| Fake News Precision | 99% |
| Fake News Recall | 98% |
| Fake News F1-score | 98% |
| Genuine News Precision | 98% |
| Genuine News Recall | 99% |
| Genuine News F1-score | 99% |

**Confusion Matrix:**

| | Predicted Fake | Predicted Genuine |
|---|---:|---:|
| Actual Fake | 3415 | 76 |
| Actual Genuine | 30 | 4209 |

*Note: These results represent the original test split. High test accuracy does not guarantee real-world performance. Duplicate articles, source-specific patterns and dataset leakage should be checked during further validation.*

---

## ✨ Key Features

- User-friendly Streamlit interface.
- News article text input.
- Automated text preprocessing.
- Fake and genuine news classification.
- Model confidence score.
- Uncertain prediction warning.
- Real-time prediction through the trained model.

---

## 📂 Project Structure

```text
fake-news-detection-system/
│
├── app.py
├── train.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── model/
│   └── fake_news_model.pkl
│
├── Fake.csv
└── True.csv
```

The CSV datasets are excluded from the public Git repository. Obtain the datasets from their original source and place them in the project directory before training.

---

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/shani-0001/fake-news-detection-system.git
```

### Step 2: Open Project Directory

```bash
cd fake-news-detection-system
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare the Dataset

Place the following files in the project directory:

- Fake.csv
- True.csv

Use the dataset format expected by `train.py`.

### Step 6: Train the Model

```bash
python train.py
```

The trained model should be saved at:

```text
model/fake_news_model.pkl
```

### Step 7: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---

## 🧪 Testing

The application has been tested using:

- Fake news dataset examples.
- Genuine news dataset examples.
- New user-provided news articles.
- Empty input validation.
- Short text validation.
- Prediction confidence display.

---

## 🚀 Future Scope

- Improve generalization using independent datasets.
- Experiment with advanced NLP models such as BERT.
- Add explainable AI techniques.
- Integrate reliable external fact-checking sources.
- Improve model calibration and uncertainty handling.
- Deploy the application online.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Python programming.
- Data preprocessing.
- Natural Language Processing.
- TF-IDF feature extraction.
- Machine Learning model training.
- Model evaluation.
- Streamlit application development.
- End-to-end ML project development.

---

## 👨‍💻 Author

Shani Yadav

GitHub:https://github.com/Shani-0001

---

## 📜 Disclaimer

This project is developed for educational and demonstration purposes. Predictions are based on patterns learned from the training dataset and should not be treated as proof that a news article is true or false. Always verify important information through reliable independent sources.

---

⭐ If you find this project useful, consider giving the repository a star!
