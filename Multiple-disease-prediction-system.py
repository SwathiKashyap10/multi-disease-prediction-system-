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
        Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        SkinThickness = st.number_input("Skin Thickness", min_value=0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function",min_value=0.000,format="%.3f")
        
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0)
        Insulin = st.number_input("Insulin Level", min_value=0)
        Age = st.number_input("Age", min_value=1, step=1)
        
    with col3:
        BloodPressure = st.number_input("Blood Pressure", min_value=0)
        BMI = st.number_input("BMI", min_value=0.00 ,format="%.2f")
        
        
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
        age = st.number_input("Age", min_value=1)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        restecg = st.number_input("Rest ECG (1 = Abnormal, 0 = Normal)", min_value=0, max_value=1)
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0,format="%.2f")
        thal = st.number_input("Thalassemia (0–2)", min_value=0, max_value=2)
        
    with col2:
        sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0, max_value=1)
        chol = st.number_input("Serum Cholesterol", min_value=0)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
        slope = st.number_input("Slope (0–2)", min_value=0, max_value=2)
        
    with col3:
        cp = st.number_input("Chest Pain Type (1–3)", min_value=1, max_value=3)
        fbs = st.number_input("Fasting Blood Sugar (1 = T, 0 = F)", min_value=0, max_value=1)
        exang = st.number_input("Exercise Induced Angina (1 = T, 0 = F)", min_value=0, max_value=1)
        ca = st.number_input("Number of Major Vessels (0–2)", min_value=0, max_value=2)
        

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
      MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=0.00000, format="%.5f")
      MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.00000, format="%.5f")
      MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.00000, format="%.5f")
      MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.00000, format="%.5f")
      MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.00000, format="%.5f")
      HNR = st.number_input("HNR", min_value=0.00000, format="%.5f")
      spread1 = st.number_input("Spread1",max_value=-0.000000,format="%.6f")
      PPE = st.number_input("PPE", min_value=0.000000, format="%.6f")

    with col2:
      MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", min_value=0.00000, format="%.5f")
      MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.00000, format="%.5f")
      Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.00000, format="%.5f")
      Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.00000, format="%.5f")
      Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.00000, format="%.5f")
      RPDE = st.number_input("RPDE", min_value=0.000000, format="%.6f")
      spread2 = st.number_input("Spread2", min_value=0.000000, format="%.6f")

    with col3:
      MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", min_value=0.00000, format="%.5f")
      MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.00000, format="%.5f")
      MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.00000, format="%.5f")
      Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.00000, format="%.5f")
      NHR = st.number_input("NHR", min_value=0.00000, format="%.5f")
      DFA = st.number_input("DFA", min_value=0.000000, format="%.6f")
      D2 = st.number_input("D2", min_value=0.000000, format="%.6f")
 
        
    
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
      radius_mean = st.number_input("Radius Mean", min_value=0.00,format="%.2f")
      area_mean = st.number_input("Area Mean", min_value=0.00,format="%.2f")
      concavity_mean = st.number_input("Concavity Mean", min_value=0.00000, format="%.5f")
      fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.00000, format="%.5f")
      perimeter_se = st.number_input("Perimeter SE", min_value=0.000,format="%.3f")
      compactness_se = st.number_input("Compactness SE", min_value=0.000000, format="%.6f")
      symmetry_se = st.number_input("Symmetry SE", min_value=0.00000, format="%.5f")
      texture_worst = st.number_input("Texture Worst", min_value=0.00, format="%.2f")
      smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0000, format="%.4f")
      concave_points_worst = st.number_input("Concave Points Worst", min_value=0.00000, format="%.5f")

    
    with col2:
      texture_mean = st.number_input("Texture Mean", min_value=0.00,format="%.2f")
      smoothness_mean = st.number_input("Smoothness Mean", min_value=0.00000, format="%.5f")
      concave_points_mean = st.number_input("Concave Points Mean", min_value=0.00000, format="%.5f")
      radius_se = st.number_input("Radius SE", min_value=0.0000, format="%.4f")
      area_se = st.number_input("Area SE", min_value=0.00,format="%.2f")
      concavity_se = st.number_input("Concavity SE", min_value=0.00000, format="%.5f")
      fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.000000, format="%.6f")
      perimeter_worst = st.number_input("Perimeter Worst", min_value=0.00,format="%.2f")
      compactness_worst = st.number_input("Compactness Worst", min_value=0.0000, format="%.4f")
      symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0000,format="%.4f")

        
    with col3:
      perimeter_mean = st.number_input("Perimeter Mean", min_value=0.00,format="%.2f")
      compactness_mean = st.number_input("Compactness Mean", min_value=0.00000, format="%.5f")
      symmetry_mean = st.number_input("Symmetry Mean", min_value=0.00000, format="%.5f")
      texture_se = st.number_input("Texture SE", min_value=0.0000, format="%.4f")
      smoothness_se = st.number_input("Smoothness SE", min_value=0.000000, format="%.6f")
      concave_points_se = st.number_input("Concave Points SE", min_value=0.000000,format="%.6f")
      radius_worst = st.number_input("Radius Worst", min_value=0.00,format="%.2f")
      area_worst = st.number_input("Area Worst", min_value=0.00,format="%.2f")
      concavity_worst = st.number_input("Concavity Worst", min_value=0.00000, format="%.5f")
      fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.00000, format="%.5f")

    

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
      age = st.number_input("Age (Years)", min_value=1)
      bp = st.number_input("Blood Pressure (mm/Hg)", min_value=0.00,format="%.2f")
      sg = st.number_input("Specific Gravity", min_value=0.000, format="%.3f")
      al = st.number_input("Albumin", min_value=0.0 ,format="%.2f")
      su = st.number_input("Sugar", min_value=0.0,format="%.2f")
      pcv = st.number_input("Packed Cell Volume", min_value=0)
      htn = st.selectbox("Hypertension", ["yes", "no"])
      appet = st.selectbox("Appetite", ["good", "poor"])

    
    with col2:
      rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
      pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
      pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
      ba = st.selectbox("Bacteria", ["present", "notpresent"])
      bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=0)
      wbcc = st.number_input("White Blood Cell Count (cells/cmm)", min_value=0)
      dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
      pe = st.selectbox("Pedal Edema", ["yes", "no"])

    
    with col3:
      bu = st.number_input("Blood Urea (mg/dl)", min_value=0)
      sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.00, format="%.2f")
      sod = st.number_input("Sodium (mEq/L)", min_value=0.00,format="%.2f")
      pot = st.number_input("Potassium (mEq/L)", min_value=0.00, format="%.2f")
      hemo = st.number_input("Hemoglobin (gms)", min_value=0.00, format="%.2f")
      rbcc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.000001, format="%.2f")
      cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
      ane = st.selectbox("Anemia", ["yes", "no"])

    
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

   