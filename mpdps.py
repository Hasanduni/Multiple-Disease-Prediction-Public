# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:21:48 2024

@author: Uvi
"""


import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Loading saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Sidebar for navigation using streamlit(it has great UI)
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System ML',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin level value')
        Age = st.text_input('Age of the Person')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
        BMI = st.text_input('BMI value')

    # Prediction button
    if st.button('Diabetes Test Result'):
        try:
            input_data = [[float(Pregnancies), float(Glucose), float(BloodPressure),
                           float(SkinThickness), float(Insulin), float(BMI),
                           float(DiabetesPedigreeFunction), float(Age)]]

            diab_prediction = diabetes_model.predict(input_data)

            if diab_prediction[0] == 1:
                st.success('The Person is Diabetic')
            else:
                st.success('The person is not Diabetic')

        except ValueError:
            st.error('Please enter valid numerical inputs.')

# Heart disease prediction page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        cp = st.text_input('Chest Pain types')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting Electrocardiographic results')
        exang = st.text_input('Exercise Induced Angina')
        oldpeak = st.text_input('ST depression induced by exercise')
        ca = st.text_input('Major vessels colored by flourosopy')
    with col2:
        sex = st.text_input('Sex')
        chol = st.text_input('Serum Cholestoral in mg/dl')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        thalach = st.text_input('Maximum Heart Rate achieved')
        slope = st.text_input('Slope of the peak exercise ST segment')
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Prediction button
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                          float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                          float(ca), float(thal)]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                st.success('The person is having heart disease')
            else:
                st.success('The person does not have any heart disease')

        except ValueError:
            st.error('Please enter valid numerical inputs.')

# Parkinson's disease prediction page
elif selected == 'Parkinsons Prediction':
    st.title('Parkinsons Disease Prediction using ML')

    # Input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        Shimmer = st.text_input('MDVP:Shimmer')
        APQ = st.text_input('MDVP:APQ')
        RPDE = st.text_input('RPDE')
        spread1 = st.text_input('spread1')
        D2 = st.text_input('D2')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        DDA = st.text_input('Shimmer:DDA')
        DFA = st.text_input('DFA')
        spread2 = st.text_input('spread2')
        PPE = st.text_input('PPE')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ3 = st.text_input('Shimmer:APQ3')
        NHR = st.text_input('NHR')
        HNR = st.text_input('HNR')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        APQ5 = st.text_input('Shimmer:APQ5')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    # Prediction button
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")

        except ValueError:
            st.error('Please enter valid numerical inputs.')
