import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Dataset  path
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Dataset load 
df = pd.read_csv(file_path)

# First 5 rows print 
print(df.head())
# Dataset ka size
print("Shape of Dataset:")
print(df.shape)

print("\n---------------------------")

# Dataset  information
print("Dataset Information:")
print(df.info())
print("\n=========================\n")

print("Missing Values:")
print(df.isnull().sum())
print("\n=========================\n")

print("Columns in Dataset:")
print(df.columns)
print("\n=========================\n")

print("Statistical Summary:")
print(df.describe())
print("\n=========================\n")

print("Churn Value Counts:")
print(df["Churn"].value_counts())

print("\nCreating Churn Graph...")

plt.figure(figsize=(6,4))

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()
print("\n==========================")
print("Removing customerID column...")
print("==========================")

df = df.drop("customerID", axis=1)

print("\nColumns after removing customerID:\n")
print(df.columns)
print("\n==========================")
print("Data Types After Removing customerID")
print("==========================")

print(df.dtypes)
print("\n==========================")
print("Encoding Churn Column")
print("==========================")

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print(df["Churn"].head())
print("\n==========================")
print("Categorical Columns")
print("==========================")

categorical_columns = df.select_dtypes(include=["object", "string"]).columns

print(categorical_columns)
print("\n==========================")
print("Encoding Gender Column")
print("==========================")

df["gender"] = df["gender"].map({
    "Male": 1,
    "Female": 0
})

print(df["gender"].head())
print("\n==========================")
print("Encoding Binary Columns")
print("==========================")

binary_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

for column in binary_columns:
    df[column] = df[column].map({
        "Yes": 1,
        "No": 0
    })

print(df[binary_columns].head())
print("\n==========================")
print("First 10 TotalCharges Values")
print("==========================")

print(df["TotalCharges"].head(10))
print("\n==========================")
print("Converting TotalCharges to Numeric")
print("==========================")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
print(df["TotalCharges"].dtype)

print("\nMissing Values in TotalCharges:")
print(df["TotalCharges"].isnull().sum())
print(df["TotalCharges"].dtype)
print("\n==========================")
print("Removing Missing Values")
print("==========================")

df = df.dropna()

print("Dataset Shape After Removing Missing Values:")
print(df.shape)
print("\n==========================")
print("One-Hot Encoding: InternetService")
print("==========================")

internet_dummies = pd.get_dummies(
    df["InternetService"],
    prefix="InternetService"
)

print(internet_dummies.head())
print("\n==========================")
print("Adding InternetService Columns")
print("==========================")

df = pd.concat([df, internet_dummies], axis=1)

print(df.head())
print("\n==========================")
print("Removing Original InternetService Column")
print("==========================")

df = df.drop("InternetService", axis=1)

print(df.columns)
print("\n==========================")
print("Columns for One-Hot Encoding")
print("==========================")

one_hot_columns = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

print(one_hot_columns)
print("\n==========================")
print("Applying One-Hot Encoding")
print("==========================")

df = pd.get_dummies(
    df,
    columns=one_hot_columns
)

print(df.shape)
print("\n==========================")
print("Creating Features and Target")
print("==========================")

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("\n==========================")
print("Train Test Split")
print("==========================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)
print("\n==========================")
print("Creating Logistic Regression Model")
print("==========================")

model = LogisticRegression(max_iter=1000)

print(model)
print("\n==========================")
print("Training Logistic Regression Model")
print("==========================")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")
print("\n==========================")
print("Making Predictions")
print("==========================")

predictions = model.predict(X_test)

print("First 10 Predictions:")
print(predictions[:10])
print("\n==========================")
print("Calculating Accuracy")
print("==========================")

accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", accuracy)
print("\n==========================")
print("Confusion Matrix")
print("==========================")

cm = confusion_matrix(y_test, predictions)

print(cm)
print("\n==========================")
print("Classification Report")
print("==========================")

report = classification_report(y_test, predictions)

print(report)
print("\n==========================")
print("Creating Decision Tree Model")
print("==========================")

dt_model = DecisionTreeClassifier(random_state=42)

print(dt_model)
print("\n==========================")
print("Training Decision Tree")
print("==========================")

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)
print("\n==========================")
print("Creating Random Forest Model")
print("==========================")

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("\n==========================")
print("Final Model Comparison")
print("==========================")

print(f"Logistic Regression : {accuracy:.4f}")
print(f"Decision Tree       : {dt_accuracy:.4f}")
print(f"Random Forest       : {rf_accuracy:.4f}")

if accuracy > dt_accuracy and accuracy > rf_accuracy:
    print("\nBest Model: Logistic Regression")
elif dt_accuracy > rf_accuracy:
    print("\nBest Model: Decision Tree")
else:
    print("\nBest Model: Random Forest")
    print("\n==========================")
print("Saving Model")
print("==========================")

joblib.dump(model, "model.pkl")

print("Model saved successfully!")
print("\n==========================")
print("Feature Names")
print("==========================")
print(X.columns.tolist())