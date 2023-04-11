import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import pickle
import streamlit as sl
from streamlit_option_menu import option_menu
with sl.sidebar:
    selected = option_menu('Machine Learning Programs', ['Insurance Prediction',"Bank Account Prediction"], icons=["shield-check","credit-card-fill"], default_index=0)
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
    
    insurance_prediction_output = ""
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
    sl.image('bankaccount.png', width=400)
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
            'location_type': location_type,
            'country' : country,
            'gender_of_respondent' : gender_of_respondent,
            'cellphone_access': cellphone_access,
            'household_size' : household_size,
            'age_of_respondent' : age_of_respondent,
            'relationship_with_head' : relationship_with_head,
            'marital_status' : marital_status,
            'education_level' : education_level,
            'job_type' : job_type,
            'year' : year,
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
    if sl.button('Bank Account Status'):
        finance_prediction = model2.predict(datadf)
        if finance_prediction[0] == 0:
            finance_prediction_output = f"The bank account status is predicted to be {finance_prediction} which states there's an account"
        if finance_prediction[0] == 1:
            finance_prediction_output = f"The bank account status is predicted to be {finance_prediction} which states there's an account"

# [[YearOfObservation,Insured_Period,Residential,Building_Painted,Building_Fenced,Garden,Settlement,Building_Dimension,Building_Type,Date_of_Occupancy,NumberOfWindows,Geo_Code]]
    sl.success(finance_prediction_output)
