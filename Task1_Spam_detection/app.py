import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict_sms(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

if __name__ == "__main__":
    while True:
        sms = input("Enter an SMS to check (or type 'quit' to exit): ")
        if sms.lower() == "quit":
            break
        print("Prediction:", predict_sms(sms))
