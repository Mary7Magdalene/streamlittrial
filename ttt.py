
import streamlit as st
import pandas as pd
import pickle as pk

# Load trained model and encoder
model = pk.load("c:/Users/Maggie/Desktop/streamlitT/Trained_model.pkl")
encoder = pk.load("c:/Users/Maggie/Desktop/streamlitT/encoder.pkl")

# Streamlit UI
st.title("ML Model Deployment with Streamlit 🚀")

st.sidebar.header("Input Features")
Pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.slider("Age", 1, 100, 25)
Fare = st.sidebar.number_input("Fare", min_value=0.0, value=10.0)
Ticket = st.sidebar.number_input("Ticket")
Embarked  = st.sidebar.selectbox("Embarked",["C","S","Q"])
Parch = st.sidebar.number_input("Parch")
SibSP = st.sidebar.number_input("SibSp")

# Convert categorical feature
encoded_sex = encoder.transform([[Sex],[Pclass],[Embarked],[Ticket]]).toarray()

# Create DataFrame
input_data = pd.DataFrame([[Pclass, Age, Fare] + encoded_sex.tolist()[0]], 
                          columns=['Pclass', 'Age', 'Fare','Parch','SibSp'] + list(encoder.get_feature_names_out(['Sex','Pclass','Embarked','Ticket'])))

# Make prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write("### Prediction: **Survived**" if prediction == 1 else "### Prediction: **Did Not Survive**")
