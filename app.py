import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

model = joblib.load("model.pkl")

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details and click Predict.")

# -----------------------
# INPUTS
# -----------------------

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0,1])

partner = st.selectbox("Partner", [0,1])

dependents = st.selectbox("Dependents", [0,1])

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=100,
    value=12
)

phone = st.selectbox(
    "Phone Service",
    [0,1]
)

paperless = st.selectbox(
    "Paperless Billing",
    [0,1]
)

monthly = st.number_input(
    "Monthly Charges",
    value=50.0
)

total = st.number_input(
    "Total Charges",
    value=600.0
)

internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

multiple = st.selectbox(
    "Multiple Lines",
    [
        "No",
        "No phone service",
        "Yes"
    ]
)

security = st.selectbox(
    "Online Security",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

backup = st.selectbox(
    "Online Backup",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

device = st.selectbox(
    "Device Protection",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

support = st.selectbox(
    "Tech Support",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

tv = st.selectbox(
    "Streaming TV",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

movies = st.selectbox(
    "Streaming Movies",
    [
        "No",
        "No internet service",
        "Yes"
    ]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# -----------------------
# BUTTON
# -----------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    input_data.loc[0,"gender"] = 1 if gender=="Male" else 0
    input_data.loc[0,"SeniorCitizen"] = senior
    input_data.loc[0,"Partner"] = partner
    input_data.loc[0,"Dependents"] = dependents
    input_data.loc[0,"tenure"] = tenure
    input_data.loc[0,"PhoneService"] = phone
    input_data.loc[0,"PaperlessBilling"] = paperless
    input_data.loc[0,"MonthlyCharges"] = monthly
    input_data.loc[0,"TotalCharges"] = total
        # -----------------------
    # Internet Service
    # -----------------------

    if internet == "DSL":
        input_data.loc[0, "InternetService_DSL"] = 1
    elif internet == "Fiber optic":
        input_data.loc[0, "InternetService_Fiber optic"] = 1
    else:
        input_data.loc[0, "InternetService_No"] = 1

    # -----------------------
    # Multiple Lines
    # -----------------------

    input_data.loc[0, f"MultipleLines_{multiple}"] = 1

    # -----------------------
    # Online Security
    # -----------------------

    input_data.loc[0, f"OnlineSecurity_{security}"] = 1

    # -----------------------
    # Online Backup
    # -----------------------

    input_data.loc[0, f"OnlineBackup_{backup}"] = 1

    # -----------------------
    # Device Protection
    # -----------------------

    input_data.loc[0, f"DeviceProtection_{device}"] = 1

    # -----------------------
    # Tech Support
    # -----------------------

    input_data.loc[0, f"TechSupport_{support}"] = 1

    # -----------------------
    # Streaming TV
    # -----------------------

    input_data.loc[0, f"StreamingTV_{tv}"] = 1

    # -----------------------
    # Streaming Movies
    # -----------------------

    input_data.loc[0, f"StreamingMovies_{movies}"] = 1

    # -----------------------
    # Contract
    # -----------------------

    input_data.loc[0, f"Contract_{contract}"] = 1

    # -----------------------
    # Payment Method
    # -----------------------

    input_data.loc[0, f"PaymentMethod_{payment}"] = 1

    # -----------------------
    # Prediction
    # -----------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Will Churn")
        st.write(f"**Probability:** {probability[1]*100:.2f}%")
    else:
        st.success("✅ Customer Will NOT Churn")
        st.write(f"**Probability:** {probability[0]*100:.2f}%")

    with st.expander("Show Input Data"):
        st.dataframe(input_data)