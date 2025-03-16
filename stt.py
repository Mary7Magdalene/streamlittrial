import pickle as pk
import pandas as pd
import numpy as np
import streamlit as st

model_path = "c:/Users/Maggie/Desktop/streamlitT/Trained_model.pkl"
data_path = "c:/Users/Maggie/Desktop/streamlitT/Titanic-Dataset.csv"

model = pk.load(open(model_path,"rb"))
data =pd.read_csv(data_path)
data = data.drop("PassengerId", axis=1)  # Make sure this line isn't before accessing it

print(data)

st.header("Titanic Survival Predictor")

Sex = st.selectbox("choose Sex",data["Sex"].unique())
if Sex == "male":
    Sex = 1
else:
    Sex = 0

Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Age = st.number_input("Passanger Age")
SibSp = st.number_input("Siblings/Spouses Aboard (SibSp)")
Parch = st.number_input("Parents/Children Aboard (Parch)")
Fare = st.number_input("Fare", )
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Button to process input
if st.button("Predict Survival"):
    st.success("Prediction functionality will be added here.")

#mapping prep
dataset_categorical =  ["Pclass", "Sex", "Embarked"]
dataset_numerical = ["Age","Survived","SibSp","Parch","Fare"]


# Create input dictionary
input_dict = {"Pclass": Pclass,"Sex": Sex,"Age": Age,"SibSp": SibSp,"Parch": Parch,"Fare": Fare,"Embarked": Embarked}

input_df = pd.DataFrame([input_dict])

#derive the dataframe of categorical features alone
categorical_df = pd.DataFrame([{col: input_dict[col] for col in dataset_categorical}])

#expansion of columns via one hot encoding
categorical_encoded_df = pd.get_dummies(categorical_df, columns=dataset_categorical)

def predictor():
    m= model.predict(categorical_encoded_df)    
    return m

predict_button = st.button('predict outcome',on_click=predictor)

if predict_button:
    result = predictor()
    st.success('The survival prediction for this passanger is{result}')


