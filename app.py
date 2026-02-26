import streamlit as st
import joblib
from train_model import TextPreprocessor


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# ---------------------------
# Email Predictor Class
# ---------------------------
class EmailPredictor:
    def __init__(self,
                 model_path="models/model.joblib",
                 vectorizer_path="models/vectorizer.joblib"):

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.text_processor = TextPreprocessor()

    def predict(self, text):
        processed_text = self.text_processor.transform_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_text)[0] 

        
        probability = self.model.predict_proba(vectorized_text).max()

        return ("Spam" if prediction == 1 else "Ham"), probability


# ---------------------------
# Cache Model Loading
# ---------------------------
@st.cache_resource
def load_predictor():
    return EmailPredictor()


# ---------------------------
# UI Section
# ---------------------------

st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check whether it is Spam or Ham.")

email_text = st.text_area("Enter Email Text", height=200)

if st.button("Check Email"):

    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        predictor = load_predictor()
        result, prob = predictor.predict(email_text)

        if result == "Spam":
            st.error(f"🚨 This email is SPAM!\n\nConfidence: {prob:.2%}")
        else:
            st.success(f"✅ This email is HAM (Not Spam).\n\nConfidence: {prob:.2%}")