import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#loading the saved model
Dscaler, diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, 'diabetes_model.sav'), 'rb')
)

# Heart
heart_disease_model = pickle.load(
    open(os.path.join(BASE_DIR, 'heart_model.sav'), 'rb')
)

# Parkinson's
Pscaler, parkinsons_model = pickle.load(
    open(os.path.join(BASE_DIR, 'parkinsons_model.sav'), 'rb')
)

# Breast Cancer
Bscaler, breast_cancer_model = pickle.load(
    open(os.path.join(BASE_DIR, 'breast_cancer_model.sav'), 'rb')
)

# CKD
Cscaler, CKD_model = pickle.load(
    open(os.path.join(BASE_DIR, 'ckd_model.sav'), 'rb')
)



def check_empty_inputs(inputs):
    return any(value.strip() == "" for value in inputs)


#sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple disease prediction system', ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Disease Prediction','Breast Cancer Prediction','CKD Prediction'],icons=['activity','heart-pulse','person','gender-female','capsule'],default_index=0)
    
    
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
        diab_user_input = [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]
        
        # 🔴 Check for empty inputs
        if check_empty_inputs(diab_user_input):
           st.error("⚠️ All input fields must be filled before prediction.")
           st.stop()
        
        try:
          diab_user_input_Final = [float(value) for value in diab_user_input]
        except:
          st.error("Please enter valid numeric values.")
          st.stop()

        
        # Scale the input before prediction
        diab_input_data_scaled = Dscaler.transform([diab_user_input_Final])
        diab_prediction = diabetes_model.predict(diab_input_data_scaled)
        
        #st.write("🔹 Raw Model Output:", diab_prediction)
        
        if diab_prediction[0] == 1:
            diab_result = 'The person is Diabetic'
            
        else:
            diab_result = 'The person is Not Diabetic'
            
        st.success(diab_result)
        
    
    
#Heart Disease prediction page
if(selected == 'Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction using ML')
    
    #getting the input fron the user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure in mm Hg')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal (0 = normal; 1 = fixed defect; 2 = reversable defect)')
        

    #code for prediction
    heart_diagnosis = ''
    
    #creating a btn for prediction
    if st.button('Heart Disease Test Result'):
        # Convert all inputs to float
        heart_user_input = [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]
        
        # 🔴 Check for empty inputs
        if check_empty_inputs(heart_user_input):
           st.error("⚠️ All input fields must be filled before prediction.")
           st.stop()
        
        try:
          heart_user_input_Final = [float(value) for value in heart_user_input]
        except:
          st.error("Please enter valid numeric values.")
          st.stop()
        

        heart_prediction = heart_disease_model.predict([ heart_user_input_Final])
        
        #st.write("🔹 Raw Model Output:", heart_prediction)
        
        if heart_prediction[0] == 1:
            heart_result = 'The person is having heart disease'
        else:
           heart_result = 'The person does not have heart disease'

        st.success(heart_result)

    
# Parkinson's Disease Prediction Page
if selected == 'Parkinsons Disease Prediction':
    # Page title
    st.title('Parkinson’s Disease Prediction using Machine Learning')

    # 3-column layout for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo_Hz = st.text_input('MDVP:Fo(Hz)')
        MDVP_Jitter_percent = st.text_input('MDVP:Jitter(%)')
        MDVP_PPQ = st.text_input('MDVP:PPQ')
        MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        MDVP_APQ = st.text_input('MDVP:APQ')
        HNR = st.text_input('HNR')
        spread1 = st.text_input('Spread1')
        PPE = st.text_input('PPE')

    with col2:
        MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz)')
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        Jitter_DDP = st.text_input('Jitter:DDP')
        Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
        Shimmer_DDA = st.text_input('Shimmer:DDA')
        RPDE = st.text_input('RPDE')
        spread2 = st.text_input('Spread2')

    with col3:
        MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz)')
        MDVP_RAP = st.text_input('MDVP:RAP')
        MDVP_Shimmer = st.text_input('MDVP:Shimmer')
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
        NHR = st.text_input('NHR')
        DFA = st.text_input('DFA')
        D2 = st.text_input('D2')
        
    
    # Prediction result variable
    parkinsons_diagnosis = ''

    # Button for prediction
    if st.button('Parkinson’s Test Result'):
            # Convert all inputs to float
            par_user_input = [
                MDVP_Fo_Hz,
                MDVP_Fhi_Hz,
                MDVP_Flo_Hz,
                MDVP_Jitter_percent,
                MDVP_Jitter_Abs,
                MDVP_RAP,
                MDVP_PPQ,
                Jitter_DDP,
                MDVP_Shimmer,
                MDVP_Shimmer_dB,
                Shimmer_APQ3,
                Shimmer_APQ5,
                MDVP_APQ,
                Shimmer_DDA,
                NHR,
                HNR,
                RPDE,
                DFA,
                spread1,
                spread2,
                D2,
                PPE
            ]
            
            # 🔴 Check for empty inputs
            if check_empty_inputs(par_user_input):
               st.error("⚠️ All input fields must be filled before prediction.")
               st.stop()
            
            try:
              par_user_input_Final = [float(value) for value in par_user_input]
            except:
              st.error("Please enter valid numeric values.")
              st.stop()

            # Scale the input data
            par_input_data_scaled = Pscaler.transform([par_user_input_Final])

            # Predict using the loaded model
            parkinsons_prediction = parkinsons_model.predict(par_input_data_scaled)

            #st.write("🔹 Raw Model Output:", parkinsons_prediction)

            # Interpretation
            if parkinsons_prediction[0] == 1:
                par_result = "The person has Parkinson’s Disease"
            else:
                par_result = "The person is Healthy"


            # Display result
            st.success(par_result)

    

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # Page title
    st.title('Breast Cancer Prediction using Machine Learning')

    # Create 3-column input layout
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        radius_mean = st.text_input('Radius Mean')
        area_mean = st.text_input('Area Mean')
        concavity_mean = st.text_input('Concavity Mean')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')
        perimeter_se = st.text_input('Perimeter SE')
        compactness_se = st.text_input('Compactness SE')
        symmetry_se = st.text_input('Symmetry SE')
        texture_worst = st.text_input('Texture Worst')
        smoothness_worst = st.text_input('Smoothness Worst')
        concave_points_worst = st.text_input('Concave Points Worst')
        
    
    with col2:
        texture_mean = st.text_input('Texture Mean')
        smoothness_mean = st.text_input('Smoothness Mean')
        concave_points_mean = st.text_input('Concave Points Mean')
        radius_se = st.text_input('Radius SE')
        area_se = st.text_input('Area SE')
        concavity_se = st.text_input('Concavity SE')
        fractal_dimension_se = st.text_input('Fractal Dimension SE')
        perimeter_worst = st.text_input('Perimeter Worst')
        compactness_worst = st.text_input('Compactness Worst')
        symmetry_worst = st.text_input('Symmetry Worst')
        
    
    with col3:
         perimeter_mean = st.text_input('Perimeter Mean')
         compactness_mean = st.text_input('Compactness Mean')
         symmetry_mean = st.text_input('Symmetry Mean')
         texture_se = st.text_input('Texture SE')
         smoothness_se = st.text_input('Smoothness SE')
         concave_points_se = st.text_input('Concave Points SE')
         radius_worst = st.text_input('Radius Worst')
         area_worst = st.text_input('Area Worst')
         concavity_worst = st.text_input('Concavity Worst')
         fractal_dimension_worst = st.text_input('Fractal Dimension Worst')
    

    # Prediction result variable
    cancer_diagnosis = ''

    # Prediction button
    if st.button('Breast Cancer Test Result'):
        # Convert all inputs to float
        bst_user_input = [
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst
        ]
        
        # 🔴 Check for empty inputs
        if check_empty_inputs(bst_user_input):
           st.error("⚠️ All input fields must be filled before prediction.")
           st.stop()
        
        try:
          bst_user_input_Final = [float(value) for value in bst_user_input]
        except:
          st.error("Please enter valid numeric values.")
          st.stop()

        # Scale the input data using the loaded scaler
        bst_input_data_scaled = Bscaler.transform([bst_user_input_Final])

        # Predict using the loaded breast cancer model
        prediction = breast_cancer_model.predict(bst_input_data_scaled)

        #st.write("🔹 Raw Model Output:", prediction)

        # Interpretation of model output
        if prediction[0] == 1:
            bst_result = 'Breast cancer is Benign'
        else:
            bst_result = 'Breast cancer is Malignant'
            

        # Display prediction result
        st.success(bst_result)
    
# CKD Prediction Page
if(selected == 'CKD Prediction'):
    
    st.title('Chronic Kidney Disease Prediction using ML')
    
    # User Input Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age (Years)')
        bp = st.text_input('Blood Pressure (mm/Hg)')
        sg = st.text_input('Specific Gravity')
        al = st.text_input('Albumin')
        su = st.text_input('Sugar')
    
    with col2:
        rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
        pc = st.selectbox('Pus Cell', ['normal', 'abnormal'])
        pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
        ba = st.selectbox('Bacteria', ['present', 'notpresent'])
        bgr = st.text_input('Blood Glucose Random (mg/dl)')
    
    with col3:
        bu = st.text_input('Blood Urea (mg/dl)')
        sc = st.text_input('Serum Creatinine (mg/dl)')
        sod = st.text_input('Sodium (mEq/L)')
        pot = st.text_input('Potassium (mEq/L)')
        hemo = st.text_input('Hemoglobin (gms)')
    
    with col1:
        pcv = st.text_input('Packed Cell Volume')
    
    with col2:
        wbcc = st.text_input('White Blood Cell Count (cells/cmm)')
    
    with col3:
        rbcc = st.text_input('Red Blood Cell Count (millions/cmm)')
    
    with col1:
        htn = st.selectbox('Hypertension', ['yes', 'no'])
    
    with col2:
        dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
    
    with col3:
        cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
    
    with col1:
        appet = st.selectbox('Appetite', ['good', 'poor'])
    
    with col2:
        pe = st.selectbox('Pedal Edema', ['yes', 'no'])
    
    with col3:
        ane = st.selectbox('Anemia', ['yes', 'no'])
    
    # Prediction Section
    ckd_result = ''
    
    if st.button('CKD Test Result'):
        
        # Encode binary & categorical values
        binary_map = {'yes':1,'no':0,'present':1,'notpresent':0,'normal':1,'abnormal':0,'good':1,'poor':0}
        
        ckd_input_data = [
            age, 
            bp,
            sg,
            al,
            su,
            binary_map[rbc],
            binary_map[pc],
            binary_map[pcc],
            binary_map[ba],
            bgr,
            bu, 
            sc,
            sod,
            pot,
            hemo,
            pcv,
            wbcc,
            rbcc,
            binary_map[htn],
            binary_map[dm],
            binary_map[cad],
            binary_map[appet],
            binary_map[pe],
            binary_map[ane]
        ]
        
        # 🔴 Check for empty inputs
        if check_empty_inputs(ckd_input_data):
           st.error("⚠️ All input fields must be filled before prediction.")
           st.stop()
        
        try:
          ckd_user_input_Final = [float(value) for value in ckd_input_data]
        except:
          st.error("Please enter valid numeric values.")
          st.stop()
        
        # Scale the input
        ckd_input_data_scaled = Cscaler.transform([ckd_user_input_Final])
        
        prediction = CKD_model.predict(ckd_input_data_scaled)
       
        #st.write("🔹 Raw Model Output:", prediction)
        
        if prediction[0] == 1:
            ckd_result = 'The patient is likely to have Chronic Kidney Disease.'
        else:
            ckd_result = 'The patient is not likely to have Chronic Kidney Disease.'
    
        st.success(ckd_result)

   