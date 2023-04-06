import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import pickle
import streamlit as sl
from streamlit_option_menu import option_menu
model = pickle.load(open('insurance.pkl', 'rb'))
selected = option_menu('Insurance Prediction Program',['Insurance Prediction'],icons=["shield-check"],default_index=0) #bootstrap icons
if (selected == "Insurance Prediction"):
    sl.title('Insurance Prediction using ML')
    sl.image('insurance.jpg')
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
