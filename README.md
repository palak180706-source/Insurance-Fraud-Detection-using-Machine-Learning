# Insurance Fraud Detection Using Machine Learning

## 📌 Project Overview
Insurance fraud is a major issue that causes significant financial losses to insurance companies every year. Detecting fraudulent claims manually is time-consuming and inefficient.

This project uses Machine Learning techniques to automatically identify fraudulent insurance claims based on historical claim data.

The system analyzes different features of insurance policies and incidents to predict whether a claim is fraudulent or genuine.

---

## 🎯 Objectives
- Detect fraudulent insurance claims using Machine Learning.
- Analyze insurance datasets to identify patterns related to fraud.
- Compare different ML algorithms to determine the best performing model.
- Deploy the model using a simple web application for prediction.

---

## 📁 Project Structure

```
Insurance-Fraud-Detection
│
├── app
│   ├── app.py
│   └── templates
│       └── index.html
│
├── data
│   └── insurance.csv
│
├── models
│   └── fraud_model.pkl
│
├── src
│   ├── train_model.py
│   └── predict.py
│
├── README.md
└── requirements.txt
```

## 📊 Dataset
The dataset contains insurance claim information including:

- Policy details  
- Customer demographics  
- Incident information  
- Claim amount  
- Vehicle details  
- Fraud report status  

**Target Variable:**  
`fraud_reported`

Values:
- **Y → Fraud**
- **N → Genuine Claim**

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Flask  
- HTML / CSS  

---

## 🤖 Machine Learning Algorithms Used
The following models were implemented and compared:

1. Decision Tree  
2. Random Forest  
3. K-Nearest Neighbors (KNN)  
4. Logistic Regression  
5. Naive Bayes  
6. Support Vector Machine (SVM)  

The best performing model is saved and used for prediction.

---

## 🔄 Project Workflow
Data Collection  
↓  
Data Cleaning  
↓  
Exploratory Data Analysis  
↓  
Feature Encoding  
↓  
Feature Scaling  
↓  
Train/Test Split  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Model Selection  
↓  
Fraud Prediction  
↓  
Web Deployment  

---

## ⚙ Model Training
This script will:

- Load the dataset  
- Preprocess the data  
- Train machine learning models  
- Evaluate model performance  
- Save the trained model in the models folder  

---

## 🌐 Running the Web Application
To start the web app:
cd app 
python app.py
Users can enter claim information and the system will predict whether it is **Fraudulent or Genuine**.

---

## 📈 Model Evaluation Metrics
The models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

These metrics help determine the best algorithm for fraud detection.

---

## 👥 Team Members
- Palak Puri (Team Lead)
- Swasti Maheshwari
- Tanya Prasad
- Prashant Kumar

---
## 🚀 Future Improvements
- Use larger real-world insurance datasets  
- Apply deep learning models  
- Improve feature engineering  
- Deploy the application on cloud platforms  

---

## 📜 License
This project is developed for educational purposes.

---

## ⭐ Conclusion
Machine Learning can significantly improve fraud detection systems by automatically identifying suspicious claims.  
This project demonstrates how different ML models can be applied to detect fraudulent insurance activities effectively.
