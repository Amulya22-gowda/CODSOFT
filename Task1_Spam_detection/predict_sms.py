import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

while True:
    user_msg = input("\nEnter an SMS to check (or type 'quit' to exit): ")
    if user_msg.lower() == 'quit':
        print("Exiting...")
        break
    msg_tfidf = vectorizer.transform([user_msg])
    prediction = model.predict(msg_tfidf)[0]
    print("Prediction:", "SPAM 🚫" if prediction == 1 else "HAM ✅")
