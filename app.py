
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and features
model = joblib.load('model.joblib')
model_features = joblib.load('model_features.joblib')

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Default Predictor")
st.markdown("Enter applicant and loan details to assess default risk.")

st.header("👤 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("Annual Income ($)", min_value=0.0, max_value=2000000.0, value=60000.0)
    debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=15.0)
    homeownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verified_income = st.selectbox("Income Verification", ["Not Verified", "Source Verified", "Verified"])
    application_type = st.selectbox("Application Type", ["individual", "joint"])

with col2:
    inquiries_last_12m = st.number_input("Credit Inquiries (Last 12 Months)", min_value=0, max_value=30, value=1)
    open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=5)
    total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0.0, value=30000.0)
    total_credit_utilized = st.number_input("Total Credit Utilized ($)", min_value=0.0, value=8000.0)
    num_historical_failed_to_pay = st.number_input("Historical Failed Payments", min_value=0, max_value=50, value=0)

st.header("💳 Loan Details")
col3, col4 = st.columns(2)

with col3:
    loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, max_value=500000.0, value=15000.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0)
    installment = st.number_input("Monthly Installment ($)", min_value=0.0, max_value=5000.0, value=350.0)
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_purpose = st.selectbox("Loan Purpose", [
        "car", "credit_card", "debt_consolidation", "educational",
        "home_improvement", "house", "major_purchase", "medical",
        "moving", "other", "renewable_energy", "small_business",
        "vacation", "wedding"
    ])

with col4:
    balance = st.number_input("Current Balance ($)", min_value=0.0, max_value=500000.0, value=10000.0)
    paid_total = st.number_input("Total Paid ($)", min_value=0.0, max_value=500000.0, value=5000.0)
    num_collections_last_12m = st.number_input("Collections Last 12 Months", min_value=0, max_value=20, value=0)
    num_cc_carrying_balance = st.number_input("Credit Cards Carrying Balance", min_value=0, max_value=30, value=2)
    account_never_delinq_percent = st.number_input("% Accounts Never Delinquent", min_value=0.0, max_value=100.0, value=90.0)

# Additional fields (with defaults)
total_credit_lines = st.number_input("Total Credit Lines", min_value=0, max_value=100, value=15)
num_satisfactory_accounts = st.number_input("Satisfactory Accounts", min_value=0, max_value=50, value=10)
num_active_debit_accounts = st.number_input("Active Debit Accounts", min_value=0, max_value=30, value=3)
total_debit_limit = st.number_input("Total Debit Limit ($)", min_value=0.0, value=5000.0)
num_total_cc_accounts = st.number_input("Total Credit Card Accounts", min_value=0, max_value=50, value=5)
num_open_cc_accounts = st.number_input("Open Credit Card Accounts", min_value=0, max_value=30, value=3)
num_mort_accounts = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=1)
tax_liens = st.number_input("Tax Liens", min_value=0, max_value=10, value=0)
public_record_bankrupt = st.number_input("Bankruptcy Records", min_value=0, max_value=10, value=0)
current_accounts_delinq = st.number_input("Current Accounts Delinquent", min_value=0, max_value=20, value=0)
accounts_opened_24m = st.number_input("Accounts Opened (Last 24 Months)", min_value=0, max_value=30, value=3)

# Encodings
homeownership_map = {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3}
verified_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
purpose_list = sorted(["car", "credit_card", "debt_consolidation", "educational",
    "home_improvement", "house", "major_purchase", "medical",
    "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"])
purpose_map = {v: i for i, v in enumerate(purpose_list)}

if st.button("🔍 Predict Default Risk"):
    input_data = pd.DataFrame([[
        annual_income, debt_to_income, inquiries_last_12m,
        total_credit_lines, open_credit_lines, total_credit_limit,
        total_credit_utilized, num_collections_last_12m,
        num_historical_failed_to_pay, current_accounts_delinq,
        accounts_opened_24m, num_satisfactory_accounts,
        num_active_debit_accounts, total_debit_limit,
        num_total_cc_accounts, num_open_cc_accounts,
        num_cc_carrying_balance, num_mort_accounts,
        account_never_delinq_percent, tax_liens,
        public_record_bankrupt, loan_amount, interest_rate,
        installment, balance, paid_total,
        homeownership_map[homeownership],
        verified_map[verified_income],
        purpose_map.get(loan_purpose, 0),
        0 if application_type == "individual" else 1,
        grade_map[grade]
    ]], columns=model_features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK: Likely to Default**")
    else:
        st.success(f"✅ **LOW RISK: Unlikely to Default**")

    st.metric("Default Probability", f"{probability:.1%}")
    st.progress(float(probability))
    st.caption("A probability above 50% indicates higher default risk.")
