Spam SMS Detection  

This project detects whether an SMS message is **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Machine Learning**.  

## 📂 Dataset  
- Dataset used: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- Stored in: `dataset/spam.csv`  

## ⚙️ Features  
- Data cleaning and preprocessing  
- TF-IDF vectorization  
- Naive Bayes classification  
- Accuracy: ~97%  
- Interactive SMS prediction  

## 📁 Project Structure  
Spam_SMS_Detection/
│── dataset/
│ └── spam.csv
│── app.py # Interactive SMS prediction script
│── predict_sms.py # Helper script for predictions
│── spam_sms_detection.py # Model training script
│── spam_model.joblib # Saved trained model
│── vectorizer.joblib # Saved TF-IDF vectorizer
│── requirements.txt # Dependencies
└── README.md # Project documentation

## 🚀 Installation & Setup  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/CODSOFT.git
   cd CODSOFT/Spam_SMS_Detection
Install dependencies:
pip install -r requirements.txt
▶️ Usage
1. Train the Model
Run the training script:
python spam_sms_detection.py
This will generate spam_model.joblib and vectorizer.joblib.

2. Predict SMS Messages
Run the app for interactive predictions:
python app.py
Example:
Enter an SMS to check (or type 'quit' to exit): Congratulations! You won free tickets
Prediction: SPAM 🚨

Enter an SMS to check (or type 'quit' to exit): Are we meeting tomorrow?
Prediction: HAM ✅
📊 Results
Algorithm: Multinomial Naive Bayes
Vectorization: TF-IDF
Accuracy: ~97%

🛠 Tech Stack
Python 🐍
Scikit-learn
Pandas, NumPy
Joblib

🙏 Acknowledgements
UCI Machine Learning Repository
Kaggle Dataset

