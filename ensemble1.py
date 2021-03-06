#preg	plas	pres	skin	insulin	mass	pedi	age	result
#import libraries
#conda install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#app heading
st.header("""
An Ensemble Machine Learning Based model for predicting predisposition to diabetic Condition
""")
st.write("""
 *Peter B. Kaaya*
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')

 #st.number_input(label, min_value=None, max_value=None, value=, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None) 



def user_input_features():
        #preg = st.sidebar.slider('Pregnancies_', 0, 10, 20)
        preg = st.sidebar.number_input('Pregnancies_', 0, 20)
        plas = st.sidebar.number_input('Glucose', 0, 200)
        pres = st.sidebar.number_input('BloodPressure', 1, 185)
        skin = st.sidebar.number_input('SkinThickness', 0.0, 110.0)
        insulin=st.sidebar.number_input('Insulin', 0.0, 900.0)
        mass=st.sidebar.number_input('BMI', 0.0, 61.5)
        pedi=st.sidebar.number_input('DiabetesPedigreeFunction', 0.0, 3.55 )
        age=st.sidebar.number_input('Age', 0, 120 )
        data = {'preg': preg,
                'plas': plas,
                'pres': pres,
                'skin': skin,
              'insulin':insulin,
              'mass':mass,
                'pedi':pedi,
                'age':age
               }
        features = pd.DataFrame(data, index=[0])
        
        
        
        return features
df = user_input_features()



st.table(df)

st.subheader('User Input parameters')
st.write(df)
#reading csv file
data=pd.read_csv("./fulltrainset.csv")
X =np.array(data[['preg', 'plas' , 'pres' , 'skin' , 'insulin' , 'mass' , 'pedi', 'age']])
Y = np.array(data['result'])
#random forest model
rfc= RandomForestClassifier()
rfc.fit(X, Y)
st.caption('Diabetic Condition Prediction Results, 1 = *POSITIVE*, 0 = *NEGATIVE*')
st.write(pd.DataFrame({
  'Result': ["NEGATIVE","POSITIVE"]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
#st.subheader('Prediction')
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)






