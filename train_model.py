import os
import nltk
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# ***** Download NLTK resources (only needed first time)
"""
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
"""


# ---------------------------
# 1️⃣ Data Loader
# ---------------------------

class DataLoader:
    def __init__(self, file_path, encoding="latin-1"):
        self.file_path = file_path
        self.encoding = encoding

    def load_data(self):
        return pd.read_csv(self.file_path, encoding=self.encoding)


# ---------------------------
# 2️⃣ Tabular Data Cleaning
# ---------------------------

class DataPreprocessor:
    def clean(self, df):
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        df = df.rename(columns={"v1": "target", "v2": "text"})
        df["target"] = df["target"].map({"spam": 1, "ham": 0})
        df = df.drop_duplicates(keep="first")
        df = df.dropna()
        return df


# ---------------------------
# 3️⃣ Text Preprocessing
# ---------------------------

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def transform_text(self, text):
        tokens = nltk.word_tokenize(text.lower())

        filtered = [
            word for word in tokens
            if word.isalnum() and word not in self.stopwords
        ]

        tagged = nltk.pos_tag(filtered)

        processed_data = []

        for word, tag in tagged:
            if tag.startswith('V'):
                pos = 'v'
            elif tag.startswith('J'):
                pos = 'a'
            elif tag.startswith('R'):
                pos = 'r'
            else:
                pos = 'n'

            processed_data.append(
                self.lemmatizer.lemmatize(word, pos)
            )

        return " ".join(processed_data)

    def transform_dataframe(self, df):
        df["transformed_text"] = df["text"].apply(self.transform_text)
        return df


# ---------------------------
# 4️⃣ Model Trainer
# ---------------------------

class ModelTrainer:
    def __init__(self, max_features=3000, test_size=0.2, random_state=2):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.model = MultinomialNB()

    def train(self, df):
        X = df["transformed_text"]
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.model.fit(X_train_tfidf, y_train)

        y_pred = self.model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        return acc, cm, precision
    

    def save_model(self, model_path="models/model.joblib", vectorizer_path="models/vectorizer.joblib"):
        # Ensure models folder exists
        os.makedirs("models", exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

        print("\nModels saved successfully...")


# ---------------------------
# 5️⃣ Execution Flow
# ---------------------------

if __name__ == "__main__":

    # Load
    loader = DataLoader("data/spam.csv")
    df = loader.load_data()

    # Clean structure
    cleaner = DataPreprocessor()
    df = cleaner.clean(df)

    # Text processing
    text_processor = TextPreprocessor()
    df = text_processor.transform_dataframe(df)

    # Train model
    trainer = ModelTrainer()
    accuracy, cm, precision = trainer.train(df)

    print("\nAccuracy:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nPrecision:", precision)

    # Save model
    trainer.save_model()