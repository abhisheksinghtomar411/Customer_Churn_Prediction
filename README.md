# 📊 Customer Churn Prediction System

A Machine Learning based web application that predicts whether a telecom customer is likely to churn using customer information.

Built using **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. This project predicts whether a customer will leave the company based on customer details such as contract type, monthly charges, tenure, internet service, payment method, and other service information.

The prediction is performed using a trained **Logistic Regression** model.

---

## ✨ Features

- Predict Customer Churn
- Interactive Streamlit Web Application
- Logistic Regression Model
- Data Preprocessing
- One-Hot Encoding
- Model Accuracy Comparison
- Decision Tree Comparison
- Random Forest Comparison
- Prediction Probability
- Clean User Interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│
├── notebooks/
│
├── src/
│
├── app.py
├── main.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/abhisheksinghtomar411/Customer_Churn_Prediction.git
```

Move into project folder

```bash
cd Customer_Churn_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Machine Learning Models

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 80.17% |
| Decision Tree | 72.56% |
| Random Forest | 77.97% |

**Best Model:** Logistic Regression

---

## 📊 Dataset

Dataset: IBM Telco Customer Churn Dataset

Features: 40

Target Variable:

- Churn = Yes
- Churn = No

---

## 📷 Application Preview

Add screenshots here.

Example:

```
images/home.png
images/prediction.png
```

---

## 👨‍💻 Author

Abhishek Tomar

GitHub

https://github.com/abhisheksinghtomar411

---

## ⭐ Future Improvements

- XGBoost
- LightGBM
- Deployment on Streamlit Cloud
- Email Notifications
- Customer Dashboard
- PDF Report Generation

---

## 📜 License

This project is created for educational purposes.
