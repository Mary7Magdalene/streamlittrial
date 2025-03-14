import pandas as pd
import streamlit as st
import joblib

#create a streamlit app title
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Input Widgets
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Button to process input
if st.button("Predict Survival"):
    st.success("Prediction functionality will be added here.")

#mapping prep
dataset = pd.read_csv("https://raw.githubusercontent.com/Mary7Magdalene/streamlittrial/refs/heads/main/Titanic-Dataset.csv")
dataset_categorical = ["Pclass", "Sex", "Embarked"]
dataset_numerical = ["Age", "SibSp", "Parch", "Fare"]



# Create input dictionary
input_dict = {"Pclass": pclass,"Sex": sex,"Age": age,"SibSp": sibsp,"Parch": parch,"Fare": fare,"Embarked": embarked}

input_df = pd.DataFrame([input_dict])

#derive the dataframe of categorical features alone
categorical_df = pd.DataFrame([{col: input_dict[col] for col in dataset_categorical}])

#expansion of columns via one hot encoding
categorical_encoded_df = pd.get_dummies(categorical_df, columns=dataset_categorical)

#import the joblib model
import os

file_path = 'logreg.joblib'
if os.path.exists(file_path):
    model = joblib.load(file_path) 
else:
    print("File does not exist.")
    

def predictor():
    m= model.predict(categorical_encoded_df)    
    return m

predict_button = st.button('predict outcome',on_click=predictor)

if predict_button:
    result = predictor()
    st.success('The survival prediction for this passanger')


# Button to process input(predict button)
if st.button("Predict Survival", key="predict_button"):
    st.write("User Input:", input_dict)
    st.write("Categorical Features DataFrame:", categorical_df)
    st.write("One-Hot Encoded DataFrame:", categorical_encoded_df)
    st.success("Prediction functionality will be added here.")
