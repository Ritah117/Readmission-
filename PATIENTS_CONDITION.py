import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.joblib")

# Set up Streamlit app
st.title("Readmission Prediction")

# Create tabs
tab1, tab2 = st.tabs(["Predict Readmission", "About"])

# Predict Readmission Tab
with tab1:
    st.header("Predict Readmission")
    
    # User inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    parity = st.number_input("Parity", min_value=0, value=0)
    
    # Example diagnosis options based on your data
    diagnosis_options = [
        "FRACTURE WRIST RADIAL/ ULNA", 
        "PROLONGED LABOUR", 
        "SEVERE ANAEMIA", 
        "MALARIA", 
        "ABORTION", 
        "HTN/URTI", 
        "S ANEMIA ? CAUSE"
    ]
    diagnosis = st.selectbox("Diagnosis", options=diagnosis_options)
    
    referring_facility = st.selectbox("Referring Facility", options=["AHERO C.H.", "AWASI MISSION", "COMMUNITY", "AHERO COUNTY HOSPITAL"])
    receiving_facility = st.selectbox("Receiving Facility", options=["JOOTRH", "AHERO C.H."])
    type_of_referral = st.selectbox("Type of Referral", options=["Emergency", "Scheduled", "Surgical", "Medical"])
    outcome = st.selectbox("Outcome", options=["Admitted", "Received", "Discharged"])

    # Prepare input data for prediction
    input_data = {
        "AGE": age,
        "SEX": 1 if sex == "Male" else 0,  # Encode Sex as 1 for Male, 0 for Female
        "PARITY": parity,
        "DIAGNOSIS": diagnosis,
        "REFERRING FACILITY": referring_facility,
        "RECEIVING FACILITY": receiving_facility,
        "TYPE OF REFERRAL": type_of_referral,
        "OUTCOME": 1 if outcome == "Admitted" else 0  # Encode outcome if necessary
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Handle one-hot encoding if applicable
    input_df_encoded = pd.get_dummies(input_df, columns=["DIAGNOSIS", "REFERRING FACILITY", "RECEIVING FACILITY", "TYPE OF REFERRAL"], drop_first=True)

    # Ensure input_df_encoded has the same columns as the model
    input_df_encoded = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df_encoded)
        st.write("Prediction:", "Readmitted" if prediction[0] == 1 else "Not Readmitted")

# About Tab
with tab2:
    st.header("About the Model")
    st.write("""
        This model is designed to predict the likelihood of patient readmission to the hospital. 
        It utilizes patient demographic information and medical history to make predictions, allowing healthcare providers 
        to identify patients who may need additional support to prevent unnecessary readmissions.
    """)
    
    st.write("### Developers:")
    st.write("1. ANNRITA MUKAMI")
    st.write("2. MARQULINE OPIYO")

    st.write("### Tools Used:")
    st.write("""
        - **Python**: Programming language used for developing the model and the web app.
        - **Streamlit**: Framework for creating interactive web applications.
        - **Scikit-learn**: Library used for building the machine learning model.
        - **Pandas**: Library for data manipulation and analysis.
        - **Joblib**: Used for saving and loading the trained model.
    """)



