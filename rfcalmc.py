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
        GN = st.sidebar.number_input('Gender', 1, 2)
        AGE = st.sidebar.number_input('Age', 0, 120)
        URINE_PROT = st.sidebar.number_input('Urine(proteinuria/ketonuria)', 0.1, 3.0)
        FBG = st.sidebar.number_input('Fast Blood Glucose', 80, 250)
        PR =st.sidebar.number_input('Heart rate/min (6-15>70-100,18)', 20, 100)
        BP_Syst =st.sidebar.number_input('BP (syst) 115-129 (mm/hg, Normal) (130-190) Abnormal Pressure', 0, 200)
        BP_Diast =st.sidebar.number_input('BP Diastolic 90-120 (mm/hg, syst/Diast)- Abnormal', 0, 70 )
        BMI=st.sidebar.number_input('Body Mass Index (weight/height) Normal 18.5 and 24.9 kg/m2', 0, 150 )
        GL=st.sidebar.number_input('Glucose Level (PLAS)138-240 diabetic 86-137 Non-Diabetic',0,150 )
        D_EX=st.sidebar.number_input('Awareness of certain foods to be avoided (2-Yes, 1-No)', 1,2)
        VAR_VAL=st.sidebar.number_input('Visual Acuity of the right / left eye', 0.0, 7.9 )
        INSULIN_HBA1c=st.sidebar.number_input('Insulin- haemoglobin level (HBA1c) LEVELS(4-5%.6%//20-38)', 0, 60)
        data = {'GN': GN,
                'AGE': AGE,
                'RINE_PROT': URINE_PROT,
                'FBG': FBG,
                'PR':PR,
                'BP_Syst': BP_Syst,
                'BP_Diast': BP_Diast,
                'BMI': BMI,
                'GL': GL,
                'D_EX':D_EX,
                'VAR_VAL': VAR_VAL,
                'INSULIN_HBA1c': INSULIN_HBA1c,
               }
        features = pd.DataFrame(data, index=[0])
        
        
        
        return features
df = user_input_features()



st.table(df)

st.subheader('User Input parameters')
st.write(df)
#reading csv file
data=pd.read_csv("./dibetes1_kaaya.csv")
X =np.array(data[['GN', 'AGE', 'URINE_PROT', 'FBG', 'PR', 'BP_Syst', 'BP_Diast', 'BMI','GL', 'D_EX', 'VAR_VAL', 'INSULIN_HBA1c']])
Y = np.array(data['DIABETES_RESULTS'])
#random forest model
rfc=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 200, 
                            max_depth=None, criterion='entropy')



rfc.fit(X, Y)
st.caption('Diabetic Condition Prediction Results, 1 = *POSITIVE*, 0 = *NEGATIVE*')
st.write(pd.DataFrame({
  'DIABETES_RESULTS': ["NEGATIVE","POSITIVE"]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
#st.subheader('Prediction')
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)






