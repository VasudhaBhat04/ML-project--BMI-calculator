import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    
    data = pd.read_csv('BMI_calculator.csv')  
    return data

# ML model training
def train_models(df):

    le_gender = LabelEncoder()
    le_bmi_category = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['BMI_Category'] = le_bmi_category.fit_transform(df['BMI_Category'])

    # Features and target for BMI Category prediction
    X = df[['Height_cm', 'Weight_kg', 'Gender', 'Age', 'BMI']]
    y_bmi = df['BMI_Category']
    y_weight_change = df['Weight_Change_Required_kg']

    # Train-test split
    X_train, X_test, y_bmi_train, y_bmi_test, y_weight_change_train, y_weight_change_test = train_test_split(X, y_bmi, y_weight_change, test_size=0.2, random_state=42)

    # RandomForest for BMI Classification
    bmi_model = RandomForestClassifier(n_estimators=100, random_state=42)
    bmi_model.fit(X_train, y_bmi_train)

    # RandomForest for Weight Change prediction
    weight_model = RandomForestRegressor(n_estimators=100, random_state=42)
    weight_model.fit(X_train, y_weight_change_train)

    return bmi_model, weight_model, le_gender, le_bmi_category

# Streamlit app interface
def main():
    
    df = load_data()


    bmi_model, weight_model, le_gender, le_bmi_category = train_models(df)

    # Streamlit input
    st.title("BMI & Weight Change Predictor")

    st.write("Enter your details:")

    # User inputs
    height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    
    gender_encoded = le_gender.transform([gender])[0]
    bmi = weight / ((height / 100) ** 2)  

    user_input = pd.DataFrame([[height, weight, gender_encoded, age, bmi]], columns=['Height_cm', 'Weight_kg', 'Gender', 'Age', 'BMI'])

    
    bmi_category = bmi_model.predict(user_input)[0]
    weight_change = weight_model.predict(user_input)[0]

    # Display predictions
    st.subheader(f"Predicted BMI Category: {le_bmi_category.inverse_transform([bmi_category])[0]}")
    st.subheader(f"Weight Change Required (in kg): {weight_change:.2f}")

# Run the app
if __name__ == "__main__":
    main()
