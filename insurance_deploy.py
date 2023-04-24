import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import pickle
import streamlit as sl
from streamlit_option_menu import option_menu
with sl.sidebar:
    selected = option_menu('Machine Learning Programs', ['Insurance Prediction',"Bank Account Prediction","Disease Prediction"], icons=["shield-check","credit-card-fill","heart-pulse-fill"], default_index=0)
    selected
if (selected == "Insurance Prediction"):
    model = pickle.load(open('insurance.pkl', 'rb'))
    sl.image('insurance.jpg', width=400)
    sl.title('Insurance Prediction using ML')
    col1, col2, col3 = sl.columns(3)
    with col1:
        YearOfObservation = sl.number_input('Year Of Observation', min_value=2000, max_value=2023, step=1)
    with col1:
        Insured_Period = sl.number_input('Period Of Insurance', min_value=0.0, max_value=1.0, step=0.01)
    with col2:
        Residential = sl.number_input('Residential Building', min_value=0, max_value=1, step=1)
    with col2:
        Building_Dimension = sl.number_input('Building Dimension', min_value=200, max_value=8000, step=100)
    with col1:
        Date_of_Occupancy = sl.number_input('Date Of Occupancy', min_value=1500, max_value=2023, step=100)
    with col2:
        NumberOfWindows = sl.number_input('Number Of Windows', min_value=0, max_value=100, step=1)
    with col1:
        Geo_Code = sl.number_input('Geo Code', min_value=1000, max_value=3000, step=100)
    with col2:
        Building_Type = sl.number_input('Building Type', min_value=0, max_value=4, step=1)
    with col3:
        Building_Painted = sl.selectbox('Building Painted', ['no', 'yes'])
    with col3:
        Building_Fenced = sl.selectbox('Building Fenced', ['no', 'yes'])
    with col3:
        Garden = sl.selectbox('Garden', ['no', 'yes'])
    with col3:
        Settlement = sl.selectbox('Settlement', ['rural', 'urban'])
    
    data = {
            'YearOfObservation': YearOfObservation,
            'Insured_Period' : Insured_Period,
            'Residential' : Residential,
            'Building_Painted': Building_Painted,
            'Building_Fenced' : Building_Fenced,
            'Garden' : Garden,
            'Settlement' : Settlement,
            'Building Dimension' : Building_Dimension,
            'Building_Type' : Building_Type,
            'Date_of_Occupancy' : Date_of_Occupancy,
            'NumberOfWindows' : NumberOfWindows,
            'Geo_Code' : Geo_Code,
             }
    

    datadf = pd.DataFrame(data, index=[0])
    if datadf['Building_Painted'].values == 'no':
        datadf[['Building_Painted_N', 'Building_Painted_V']] = [0.0,1.0]
  
    if datadf['Building_Painted'].values == 'yes':
      datadf[['Building_Painted_N', 'Building_Painted_V']] = [1.0,0.0]

    if datadf['Building_Fenced'].values == 'no':
      datadf[["Building_Fenced_N","Building_Fenced_V"]] = [1.0,0.0]

    if datadf['Building_Fenced'].values == 'yes':
      datadf[["Building_Fenced_N","Building_Fenced_V"]] = [0.0,1.0]

    if datadf['Garden'].values == 'no':
      datadf[["Garden_O","Garden_V"]] = [1.0,0.0]

    if datadf['Garden'].values == 'yes':
      datadf[["Garden_O","Garden_V"]] = [0.0,1.0]

    if datadf['Settlement'].values == 'urban':
      datadf[["Settlement_R","Settlement_U"]] = [0.0,1.0]

    if datadf['Settlement'].values == 'rural':
      datadf[["Settlement_R","Settlement_U"]] = [1.0,0.0]
    datadf = datadf.drop(columns=["Building_Painted","Building_Fenced","Garden","Settlement"],axis=1)
    scaler = StandardScaler()
    datadf[['YearOfObservation', 'Insured_Period',"Residential","Building_Type","Building Dimension","Date_of_Occupancy","NumberOfWindows","Geo_Code"]] = StandardScaler().fit_transform(datadf[['YearOfObservation', 'Insured_Period',"Residential","Building_Type","Building Dimension","Date_of_Occupancy","NumberOfWindows","Geo_Code"]])
    
    insurance_prediction_output = " "
    if sl.button('Insurance Claim'):
        insurance_prediction = model.predict(datadf)
        if insurance_prediction[0] == 0:
            insurance_prediction_output = f"The insurance claim is predicted to be {insurance_prediction} which states there's no claim"
        if insurance_prediction[0] == 1:
            insurance_prediction_output = f"The insurance claim is predicted to be {insurance_prediction} which states there's a claim"

# [[YearOfObservation,Insured_Period,Residential,Building_Painted,Building_Fenced,Garden,Settlement,Building_Dimension,Building_Type,Date_of_Occupancy,NumberOfWindows,Geo_Code]]
    sl.success(insurance_prediction_output)
if (selected == "Bank Account Prediction"):
    model2 = pickle.load(open('financial.pkl', 'rb'))
    sl.image('bankaccount.png', width=300)
    sl.title('Bank Account Prediction Using ML')
    col1, col2, col3 = sl.columns(3)
    with col1:
        year = sl.number_input('Year', min_value=2000, max_value=2023, step=1)
    with col2:
        job_type = sl.selectbox('Type of job',['Self employed', 'Government Dependent','Formally employed Private', 'Informally employed','Formally employed Government', 'Farming and Fishing','Remittance Dependent', 'Other Income','Dont Know/Refuse to answer', 'No Income'])
    with col2:
        education_level = sl.selectbox('Level Of Education',['Secondary education', 'No formal education','Vocational/Specialised training', 'Primary education','Tertiary education', 'Other/Dont know/RTA'])
    with col3:
        marital_status = sl.selectbox('Marital Status',['Married/Living together','Widowed','Single/Never Married','Divorced/Seperated','Dont know'])
    with col3:
        relationship_with_head = sl.selectbox('Relationship with head',['Spouse','Head of Household','Other relative','Child','Parent',"Other non-relatives"])
    with col1:
        age_of_respondent = sl.number_input('Age', min_value=0, max_value=100, step=1)
    with col2:
        household_size = sl.number_input('Household Size', min_value=0, max_value=30, step=1)
    with col3:
        cellphone_access = sl.selectbox('Cellphone Access', ['No', 'Yes'])
    with col1:
        gender_of_respondent = sl.selectbox('Gender', ['Female', 'Male'])
    with col1:
        country = sl.selectbox('Country', ['Kenya','Rwanda','Tanzania','Uganda'])
    with col3:
        location_type = sl.selectbox('Location Type', ['Rural', 'Urban'])
    
    data = {
        'country' : country,
        'year' : year,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size' : household_size, 
        'age_of_respondent' : age_of_respondent,
        'gender_of_respondent' : gender_of_respondent,
        'relationship_with_head' : relationship_with_head,
        'marital_status' : marital_status,
        'education_level' : education_level,
        'job_type' : job_type,
            }
    

    datadf = pd.DataFrame(data, index=[0])
    if datadf["country"].values == "Kenya":
        datadf[["country_Kenya","country_Rwanda","country_Tanzania","country_Uganda"]] = [1.0,0.0,0.0,0.0]
    if datadf["country"].values == "Rwanda":
        datadf[["country_Kenya","country_Rwanda","country_Tanzania","country_Uganda"]] = [0.0,1.0,0.0,0.0]
    if datadf["country"].values == "Tanzania":
        datadf[["country_Kenya","country_Rwanda","country_Tanzania","country_Uganda"]] = [0.0,0.0,1.0,0.0]
    if datadf["country"].values == "Uganda":
        datadf[["country_Kenya","country_Rwanda","country_Tanzania","country_Uganda"]] = [0.0,0.0,0.0,1.0]
    if datadf["location_type"].values == "Rural":
        datadf["location_type"] = [0.0]
    if datadf["location_type"].values == "Urban":
        datadf["location_type"] = [1.0]
    if datadf["cellphone_access"].values == "Yes":
        datadf["cellphone_access"] = [1.0]
    if datadf["cellphone_access"].values == "No":
        datadf["cellphone_access"] = [0.0]
    if datadf["gender_of_respondent"].values == "Female":
        datadf["gender_of_respondent"] = [0.0]
    if datadf["gender_of_respondent"].values == "Male":
        datadf["gender_of_respondent"] = [1.0]
    if datadf["relationship_with_head"].values == "Child":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [1.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["relationship_with_head"].values == "Head of Household":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [0.0,1.0,0.0,0.0,0.0,0.0]
    if datadf["relationship_with_head"].values == "Other non-relatives":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [0.0,0.0,1.0,0.0,0.0,0.0]
    if datadf["relationship_with_head"].values == "Other relative":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [0.0,0.0,0.0,1.0,0.0,0.0]
    if datadf["relationship_with_head"].values == "Parent":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [0.0,0.0,0.0,0.0,1.0,0.0]
    if datadf["relationship_with_head"].values == "Spouse":
        datadf[['relationship_with_head_Child','relationship_with_head_Head of Household','relationship_with_head_Other non-relatives','relationship_with_head_Other relative','relationship_with_head_Parent', 'relationship_with_head_Spouse']] = [0.0,0.0,0.0,0.0,0.0,1.0]
    if datadf["marital_status"].values == "Married/Living together":
        datadf[['marital_status_Divorced/Seperated','marital_status_Dont know','marital_status_Married/Living together','marital_status_Single/Never Married', 'marital_status_Widowed']] = [0.0,0.0,1.0,0.0,0.0]
    if datadf["marital_status"].values == "Divorced/Seperated":
        datadf[['marital_status_Divorced/Seperated','marital_status_Dont know','marital_status_Married/Living together','marital_status_Single/Never Married', 'marital_status_Widowed']] = [1.0,0.0,0.0,0.0,0.0]
    if datadf["marital_status"].values == "Dont know":
        datadf[['marital_status_Divorced/Seperated','marital_status_Dont know','marital_status_Married/Living together','marital_status_Single/Never Married', 'marital_status_Widowed']] = [0.0,1.0,0.0,0.0,0.0]
    if datadf["marital_status"].values == "Single/Never Married":
        datadf[['marital_status_Divorced/Seperated','marital_status_Dont know','marital_status_Married/Living together','marital_status_Single/Never Married', 'marital_status_Widowed']] = [0.0,0.0,0.0,1.0,0.0]
    if datadf["marital_status"].values == "Widowed":
        datadf[['marital_status_Divorced/Seperated','marital_status_Dont know','marital_status_Married/Living together','marital_status_Single/Never Married', 'marital_status_Widowed']] = [0.0,0.0,0.0,0.0,1.0]
    if datadf["education_level"].values == "No formal education":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [1.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["education_level"].values == "Other/Dont know/RTA":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [0.0,1.0,0.0,0.0,0.0,0.0]
    if datadf["education_level"].values == "Primary education":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [0.0,0.0,1.0,0.0,0.0,0.0]
    if datadf["education_level"].values == "Secondary education":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [0.0,0.0,0.0,1.0,0.0,0.0]
    if datadf["education_level"].values == "Tertiary education":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [0.0,0.0,0.0,0.0,1.0,0.0]
    if datadf["education_level"].values == "Vocational/Specialised training":
        datadf[['education_level_No formal education','education_level_Other/Dont know/RTA','education_level_Primary education','education_level_Secondary education','education_level_Tertiary education','education_level_Vocational/Specialised training']] = [0.0,0.0,0.0,0.0,0.0,1.0]
    if datadf["job_type"].values == "Dont Know/Refuse to answer":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Farming and Fishing":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Formally employed Government":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Formally employed Private":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Government Dependent":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Informally employed":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "No Income":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    if datadf["job_type"].values == "Other Income":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
    if datadf["job_type"].values == "Remittance Dependent":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
    if datadf["job_type"].values == "Self employed":
        datadf[['job_type_Dont Know/Refuse to answer', 'job_type_Farming and Fishing','job_type_Formally employed Government','job_type_Formally employed Private', 'job_type_Government Dependent','job_type_Informally employed', 'job_type_No Income','job_type_Other Income', 'job_type_Remittance Dependent','job_type_Self employed']] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
    datadf = datadf.drop(columns=["country","relationship_with_head","marital_status","education_level","job_type"],axis=1)
    scaler = StandardScaler()
    datadf[['year','household_size',"age_of_respondent"]] = StandardScaler().fit_transform(datadf[['year','household_size',"age_of_respondent"]])
    finance_prediction_output = ""
    with col2:
        sl.write(" ")
        sl.write(" ")
        if sl.button('Bank Account Status'):
            finance_prediction = model2.predict(datadf)
            if finance_prediction[0] == 0:
                finance_prediction_output = f"The bank account status is predicted to be {finance_prediction} which states there's no account"
            if finance_prediction[0] == 1:
                finance_prediction_output = f"The bank account status is predicted to be {finance_prediction} which states there's an account"

    sl.success(finance_prediction_output)
if (selected == "Disease Prediction"):
    sl.image('disease.png', width=400)
    sl.title('Disease Prediction From Symtoms')
    model = pickle.load(open('disease_prediction2.pkl', 'rb'))
    all_symptoms = ['abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feet', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dyschromic_patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'fluid_overload.1', 'foul_smell_of urine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'loss_of_taste', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurrying', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremities', 'swollen_legs', 'throat_irritation', 'tiredness', 'toxic_look_(typhus)', 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes',"weakness_in_limbs", "weakness_of_one_body_side", "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin"]
    general_symptoms = ['fatigue', 'malaise', 'lethargy', 'loss_of_appetite', 'weight_gain', 'weight_loss', 'fever', 'chills', 'sweating']

    gastrointestinal_symptoms = ['abdominal_pain', 'belly_pain', 'constipation', 'diarrhoea', 'nausea', 'stomach_bleeding', 'stomach_pain', 'vomiting', 'indigestion', 'pain_during_bowel_movements', 'pain_in_anal_region']

    respiratory_symptoms = ['cough', 'breathlessness', 'blood_in_sputum', 'rusty_sputum', 'mucoid_sputum']

    cardiovascular_symptoms = ['chest_pain', 'palpitations', 'fast_heart_rate', 'swollen_legs', 'swollen_extremities', 'swollen_blood_vessels', 'prominent_veins_on_calf']

    neurological_symptoms = ['headache', 'dizziness', 'loss_of_balance', 'slurred_speech', 'movement_stiffness', 'altered_sensorium', 'unsteadiness', 'visual_disturbances', 'weakness_in_limbs', 'weakness_of_one_body_side', 'neck_pain']

    musculoskeletal_symptoms = ['back_pain', 'joint_pain', 'hip_joint_pain', 'knee_pain', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'swelling_joints', 'loss_of_balance']

    skin_symptoms = ['skin_rash', 'red_spots_over_body', 'yellowish_skin', 'itching', 'skin_peeling', 'blister', 'blackheads', 'pus_filled_pimples', 'red_sore_around_nose', 'small_dents_in_nails', 'brittle_nails']

    urogenital_symptoms = ['urinary_tract_infection', 'bladder_discomfort', 'burning_micturition', 'continuous_feel_of_urine', 'foul_smell_of_urine', 'polyuria', 'spotting_urination', 'yellow_urine']

    mental_health_symptoms = ['anxiety', 'depression', 'irritability', 'mood_swings', 'lack_of_concentration', 'restlessness', 'coma']

    immune_system_symptoms = ['enlarged_thyroid', 'history_of_alcohol_consumption', 'inflammatory_nails', 'nodal_skin_eruptions', 'obesity', 'toxic_look_(typhus)']
    col1, col2, col3 = sl.columns(3)
    with col1:
        general_symptoms = sl.selectbox('GENERAL', general_symptoms)
    with col1:
        mental_health_symptoms = sl.selectbox('MENTAL HEALTH', mental_health_symptoms)
    with col1:
        neurological_symptoms = sl.selectbox('NEUROLOGICAL', neurological_symptoms)
    with col2:
        skin_symptoms = sl.selectbox('SKIN', skin_symptoms)
    with col2:
        urogenital_symptoms = sl.selectbox('UROGENITAL', urogenital_symptoms)
    with col2:
        gastrointestinal_symptoms = sl.selectbox('GASTROINTESTINAL', gastrointestinal_symptoms)
    with col2:
        immune_system_symptoms = sl.selectbox('IMMUNE SYSTEMS', immune_system_symptoms)
    with col3:
        musculoskeletal_symptoms = sl.selectbox('MUSCULAR', musculoskeletal_symptoms)
    with col3:
        cardiovascular_symptoms = sl.selectbox('CARDIOVASCULAR',cardiovascular_symptoms)    
    with col3:
        respiratory_symptoms = sl.selectbox('RESPIRATORY', respiratory_symptoms)



    symptom_dict = {}
    for symptom in all_symptoms:
        if symptom in selected_options:
            symptom_dict[symptom] = 1
        else:
            symptom_dict[symptom] = 0

    data = pd.DataFrame([symptom_dict])
    disease_prediction_output = ""
    if sl.button("Disease Prediction"):
        disease_prediction = model.predict(data)
        disease_prediction_output=disease_prediction
    sl.success(finance_prediction_output)





