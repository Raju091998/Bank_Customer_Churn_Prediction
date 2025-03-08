import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Load the model and DictVectorizer
model_file = "model_C=1.0.bin"
if not os.path.exists(model_file):
    st.error(f"Model file not found: {model_file}")
    st.stop()

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)  # Ensure the pickle file contains both DictVectorizer and Model

def main():
    # Load Images
    try:
        image = Image.open('images/icone.png')
        image2 = Image.open('images/image.png')
    except FileNotFoundError:
        st.error("One or more image files are missing. Ensure 'images/icone.png' and 'images/image.png' exist.")
        return

    # Display Images
    st.image(image, use_column_width=False)
    st.sidebar.image(image2)
    
    # Sidebar options
    st.sidebar.info('This app is created to predict Customer Churn')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch")
    )

    st.title("Predicting Customer Churn")

    if add_selectbox == 'Online':
        gender = st.selectbox('Gender:', ['male', 'female'])
        NumberofProducts = st.selectbox('Number of Products:', [1, 2, 3, 4])
        HasCrCard = st.selectbox('Does the Customer have a Credit Card (1=Yes, 0=No)?', [0, 1])
        IsActiveMember = st.selectbox('Is the Customer an Active Member (1=Yes, 0=No)?', [0, 1])
        Geography = st.selectbox('Country:', ['Germany', 'France', 'Spain'])
        Age = st.number_input('Age:', min_value=18, max_value=90, value=18)
        EstimatedSalary = st.number_input('Estimated Salary:', min_value=5000, max_value=90000, value=5000)
        CreditScore = st.number_input('Credit Score:', min_value=376, max_value=850, value=376)
        tenure = st.number_input('Tenure (months with current provider):', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly Charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges

        input_dict = {
            "gender": gender,
            "NumberofProducts": NumberofProducts,
            "Age": Age,
            "HasCrCard": HasCrCard,
            "Geography": Geography,
            "EstimatedSalary": EstimatedSalary,
            "CreditScore": CreditScore,
            "IsActiveMember": IsActiveMember,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        if st.button("Predict"):
            try:
                # Debugging: Print input dictionary
                st.write("Input Dictionary:", input_dict)

                # Transform input using DictVectorizer
                X = dv.transform([input_dict])
                st.write("Transformed Input (Sparse Matrix):", X)  # Debugging

                # Predict probability
                y_pred = model.predict_proba(X)[0, 1]
                st.write("Predicted Probability:", y_pred)  # Debugging

                # Determine churn
                churn = y_pred >= 0.5
                output_prob = float(y_pred)
                output = bool(churn)

                # Display results
                st.success(f'Churn: {output}, Risk Score: {output_prob:.2f}')
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    elif add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        if file_upload is not None:
            try:
                # Read CSV file
                data = pd.read_csv(file_upload)
                st.write("Uploaded Data:", data)  # Debugging

                # Convert to dictionary
                data_dict = data.to_dict(orient='records')
                st.write("Data Dictionary:", data_dict)  # Debugging

                # Transform data using DictVectorizer
                X = dv.transform(data_dict)
                st.write("Transformed Data (Sparse Matrix):", X)  # Debugging

                # Predict probabilities
                y_pred = model.predict_proba(X)[:, 1]
                churn_results = (y_pred >= 0.5).astype(bool)

                # Add predictions to the dataframe
                data["Churn Prediction"] = churn_results
                data["Risk Score"] = y_pred

                # Display results
                st.write("Predictions:", data)
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == '__main__':
    main()