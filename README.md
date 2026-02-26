# 📧 SMS Spam Classifier Web App

A Machine Learning powered SMS/Email Spam Classification Web Application built using **Python, Scikit-learn, NLTK, and Streamlit**.

This project demonstrates a complete ML workflow:
- Data Cleaning
- Text Preprocessing (NLP)
- Feature Engineering (TF-IDF)
- Model Training (Multinomial Naive Bayes)
- Model Persistence (Joblib)
- Web Deployment using Streamlit

---

## 🚀 Project Overview

Spam detection is a classic Natural Language Processing (NLP) problem.  
This project classifies messages as:

- ✅ **Ham (Not Spam)**
- 🚨 **Spam**

The model is trained using the SMS Spam dataset and deployed as an interactive web application.

---

## 🏗 Project Structure
```text
sms_spam_web_app/
│
├── app.py # Streamlit web application
├── predict.py # Standalone prediction script
├── train_model.py # Model training pipeline
│
├── data/
│ └── spam.csv # Dataset
│
├── models/
│ ├── model.joblib # Trained Naive Bayes model
│ └── vectorizer.joblib # TF-IDF vectorizer
│
├── eda/
│ ├── eda.py / eda.ipynb # Exploratory Data Analysis
│ └── plots/ # Saved visualization images
│
├── assets/ # UI images (optional)
│
└── README.md
```
---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Cleaning
- Removed unnecessary columns
- Renamed columns
- Encoded target labels (spam → 1, ham → 0)
- Removed duplicates
- Dropped missing values

### 2️⃣ Text Preprocessing (NLP)
- Lowercasing
- Tokenization (NLTK)
- Stopword removal
- POS tagging
- Lemmatization

### 3️⃣ Feature Engineering
- TF-IDF Vectorization
- Max features: 3000

### 4️⃣ Model
- **Multinomial Naive Bayes**
- Stratified train-test split
- Evaluation metrics:
  - Accuracy
  - Precision
  - Confusion Matrix

---

## 📊 Model Performance

Example metrics (may vary slightly):

- Accuracy: ~97.3%
- Precision (Spam): 0.99

- Confusion Matrix
- [[902   1]
- [ 26 105]]

---

## 🛠 Installation

### 1️⃣ Clone Repository 
```text
git clone <your-repo-link>
cd sms_spam_web_app
```

### 2️⃣ Create Virtual Environment (Recommended)
```text
python -m venv myvenv
source myvenv/bin/activate   # Linux / Mac
```

### 3️⃣ Install Dependencies
```text
pip install -r requirements.txt
```

## 📦 Download NLTK Resources (Run Once)
```text
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
```


## 🏋️ Train the Model
```text
python train_model.py
```

This will:
1) Train the model
2) Save model files inside models/

## 🔍 Run Prediction Script
```text
python predict.py
```

## 🌐 Run Web Application
```text
streamlit run app.py
```

## 📈 Future Improvements
1. Add Cross-Validation
2. Add GridSearchCV
3. Improve feature engineering
4. Add Docker support
5. Deploy to cloud (Render / Railway / Streamlit Cloud)
6. Add logging and model monitoring


## 🎯 Key Skills Demonstrated

1. Object-Oriented Programming (OOP)
2. NLP preprocessing pipeline
3. TF-IDF vectorization
4. Naive Bayes classification
5. Model persistence
6. Streamlit web deployment
7. Project structuring best practices

## 📜 License
This project is for educational and portfolio purposes.