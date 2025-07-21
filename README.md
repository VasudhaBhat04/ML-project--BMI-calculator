## BMI Calculator using Machine Learning (Random Forest)
This project is a simple and interactive BMI Calculator powered by Machine Learning. It predicts a person's BMI Category (like Underweight, Normal, Overweight, or Obese) and also suggests the required weight change (in kg) to reach a healthy BMI.

Built with a Random Forest algorithm and deployed using Streamlit, this tool provides an easy way to evaluate your BMI and understand your health status.
 Dataset Details
The dataset contains the following columns:

Height_cm: Height of the individual (in centimeters)

Weight_kg: Weight of the individual (in kilograms)

Gender: Gender of the individual (Male/Female)

Age: Age (18–100 years)

BMI: Calculated as weight (kg) / height (m²)

BMI_Category: Category label - Underweight, Normal, Overweight, or Obese

Weight_Change_Required_kg: Suggested weight change (positive or negative) to achieve a healthy BMI

Model Info
Algorithm: Random Forest Classifier

Task: Multi-class classification (BMI Category) and regression (Weight change)

Trained on: Cleaned BMI dataset with labeled outputs
