import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
#Application title
st.title('HDB Resale Price Prediction')

#define options for user inputs
towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']

#user inputs
town_selected = st.selectbox('Select Town', towns)
flat_type_selected = st.selectbox('Select Flat Type', flat_types)
storey_range_selected = st.selectbox('Select Storey', storey_ranges)
floor_area_selected = st.slider('Enter Floor Area (sqm)', min_value=30, max_value=200, value=70)

#predict button
if st.button('Predict price'):
    #create a dict for input data
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area_sqm': floor_area_selected
    }

    #convert input data to DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected], 
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area_selected]
    })

    ##One hot encoding
    df_input = pd.get_dummies(df_input, columns=['town', 'flat_type', 'storey_range'])
    
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    
    #predict the price
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f'The predicted resale price is ${y_unseen_pred:,.2f}')


st.markdown(
    f"""
    <style>
        .stApp {{
            background: url("https://tse1.mm.bing.net/th/id/OIP.6gma92EKqQaMF6B6EBjMzwHaE7?r=0&rs=1&pid=ImgDetMain&o=7&rm=3");
            background-size: cover;
            }}
    </style>
    """, unsafe_allow_html=True)