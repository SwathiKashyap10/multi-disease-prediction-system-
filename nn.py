# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved model
scaler,diabetes_model = pickle.load(open('/Users/swathikashyap/Desktop/multiple-disease-prediction-system/saved models/diabetes_model.sav','rb'))

heart_disease_model = pickle.load(open('/Users/swathikashyap/Desktop/multiple-disease-prediction-system/saved models/heart_model.sav','rb'))

parkinsons_model = pickle.load(open('/Users/swathikashyap/Desktop/multiple-disease-prediction-system/saved models/parkinsons_model.sav','rb'))

breast_cancer_model = pickle.load(open('/Users/swathikashyap/Desktop/multiple-disease-prediction-system/saved models/breast_cancer_model (1).sav','rb'))


#sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple disease prediction system', ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Disease Prediction','Breast Cancer Prediction'],icons=['activity','heart','person','gender-female'],default_index=0)
    
    
#Diabetes prediction page
if(selected == 'Diabetes Prediction'):
    #page title
    st.title('Diabetes Prediction using ML')
    
    #getting the input fron the user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number Of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
        
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
        
    with col3:
        BMI = st.text_input('BMI Value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        
    with col2:
        Age = st.text_input('Age Of The Person')
        
    #code for prediction
    diab_diagnosis = ''
    
    #creating a btn for prediction
    if st.button('Diabetes Test Result'):
        # Convert all inputs to float
        user_input = [
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]
        
        # Scale the input before prediction
        input_data_scaled = scaler.transform([user_input])
        diab_prediction = diabetes_model.predict(input_data_scaled)
        
        st.write("🔹 Raw Model Output:", diab_prediction)
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is Diabetic'
            
        else:
            diab_diagnosis = 'The person is Not Diabetic'
            
    st.success(diab_diagnosis)
        
    
if(selected == 'Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction using ML')
    
if(selected == 'Parkinsons Disease Prediction'):
    #page title
    st.title('Parkinsons Disease Prediction using ML')
    
if(selected == 'Breast Cancer Prediction'):
    #page title
    st.title('Breast Cancer Prediction using ML')