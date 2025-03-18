import pandas as pd
import pickle
import streamlit as st
import numpy as np
import os


# Load the model using a relative path
model_path = os.path.join(os.path.dirname(r"c:\Users\Maggie\Desktop\streamlitT\Trained_model.pkl"), 'Trained_model.pkl')
model = pickle.load(open(model_path, "rb"))

#main function
def main():
    st.title("Titanic Survival Prediction")
    st.image(r"c:\Users\Maggie\Desktop\streamlitT\titanic_sinking.jpeg", caption="Sinking of the Titanic RMS: 15 April 1912 in North Atlantic Ocean", use_container_width=True)
    st.write("""## Would you have survived from the Titanic disaster?""")

    # Sidebar for configurations
    st.sidebar.header("More details:")
    st.sidebar.markdown("[For more facts about the Titanic here](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic)")
    st.sidebar.markdown("----- Check your Survival Chances -----")

    # The UI framework
    Age = st.slider("Enter Age:", 1, 75, 30)
    Fare = st.slider("Fare in 1912 (in dollars $):", 15, 500, 40)
    SibSp = st.selectbox("How many siblings or spouses are you traveling with?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    Parch = st.selectbox("How many parents or children are you traveling with?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    Sex = st.selectbox("Select gender:", ["male", "female"])
    Sex = 0 if Sex == "male" else 1
    Pclass = st.selectbox("Select passenger class:", [1, 2, 3])
    Ticket = st.number_input("Enter the ticket number")
    Embarked = st.selectbox("Select the area where to board the Titanic:", ['C','Q','S'])
    
    if Embarked == 'C':
        Embarked =1
    elif Embarked == 'Q':
        Embarked = 0
    else:
        Embarked =2


    # Getting and framing data
    data = {'Age': Age, 'Fare': Fare, 'Parch': Parch, 'Sex': Sex, 'Pclass': Pclass, 'Embarked': Embarked, 'SibSp': SibSp,'Ticket':Ticket}
    df = pd.DataFrame(data, index=[0])
    return df

# Call the main function and store the returned DataFrame
data = main()


# Check if data is None or empty
if data is None or data.empty:
    st.error("No data available for prediction.")
else:
    # Prediction
    if st.button("Predict"):
        try:
            result = model.predict(data)
            proba = model.predict_proba(data)

            if result[0] == 0:
                st.success("Congratulations!!!... *You would have probably made it!*")
                st.image(r"c:\Users\Maggie\Desktop\streamlitT\lifeboat.jpeg")
                st.write("Survival probability: 'NO': {}% 'YES': {}%".format(round((proba[0, 0]) * 100, 2), round((proba[0, 1]) * 100, 2)))
            else:
                st.error("Better luck next time!!!!.... **You're probably ended like 'Jack'**")
                st.image(r"c:\Users\Maggie\Desktop\streamlitT\rip.jpeg")
                st.write("Survival probability: 'NO': {}% 'YES': {}%".format(round((proba[0, 0]) * 100, 2), round((proba[0, 1]) * 100, 2)))
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


#predict button and configuration
if st.button("working"):
   st.write("""# How's prediction working:- Insider survival facts and tips:
            -Only about '32%' of the passagers survived in this accident/n
            -Ticket price:
              1st-class:150$ -450$;2nd-class:60$;3rd-class:15$-45$/n
            -About family factor:
            if you boraded with at leas one family member '51%' survival rate
            """)
   st.image(r"c:\Users\Maggie\Desktop\streamlitT\gr.jpeg")