import joblib
from train_model import TextPreprocessor

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


email_text = "Congratulations! You have won ₹5,00,000 in the Lucky Draw. "

predictor = EmailPredictor()

result, prob = predictor.predict(email_text)

if result == "Spam":
    print(f"🚨 This email is SPAM!\n\nConfidence: {prob:.2%}")
else:
    print(f"✅ This email is HAM (Not Spam).\n\nConfidence: {prob:.2%}")