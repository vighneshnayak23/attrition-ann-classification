import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go

# Set the page configuration to 'wide' layout
st.set_page_config(layout="wide")

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the list of features in the correct order
with open('final_features.pkl', 'rb') as file:
    final_features = pickle.load(file)

data = pd.read_csv('Churn_Modelling.csv')
# st.table(data.head(2))

# Create three columns to center the title
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.title('Customer Churn Prediction', anchor='top',
             help='Predict whether a customer is likely to churn based on their profile.')

col1, col2 = st.columns([1, 1])

with col1:
    col1.subheader('Customer Profile Input')
    # User input
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', data['Gender'].unique().tolist())
    age = st.slider('Age', 18, 100)
    # balance = st.number_input('Balance')
    balance = st.slider('Balance', int(data['Balance'].min()), int(round(data['Balance'].max(), -5)), step=1000)
    credit_score = st.slider('Credit Score', 0, 1000, step=50)
    estimated_salary = st.slider('Estimated Salary', 1200, int(round(data['EstimatedSalary'].max(), -5)), step=1000)
    # estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 50)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

display_data = input_data.copy()
display_data['Geography'] = geography
display_data = display_data[['Geography'] + [col for col in display_data.columns if col != 'Geography']]


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Gender encoding
input_data['gender_male']= (input_data['Gender'] == 'Male').astype(int)
del input_data['Gender']

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data = input_data[final_features]
input_data_scaled = scaler.transform(input_data)

with col2:
    st.subheader('Input Data')
    display_data = display_data.T.reset_index()
    display_data.columns = ['Feature', 'Value']
    # Convert to DataFrame for better display
    display_data = pd.DataFrame(display_data)
    st.write(display_data)
    # Predict churn

    # Crucial Step: Re-order the DataFrame columns
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown(f'**Churn Probability: {prediction_proba:.2f}**')

    if prediction_proba > 0.5:
        st.write('**The customer is likely to churn.**')
    else:
        st.write('**The customer is not likely to churn.**')

    prob_table = pd.DataFrame({
        'Probability': [np.round(prediction_proba, 2), np.round(1 - prediction_proba, 2)]
    }, index=['Churn', 'Not Churn'])
    # st.table(prob_table)

    # col2.bar_chart(prob_table, use_container_width=True, color=['red', 'green'])
    # Custom colored bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=prob_table.index,
            y=prob_table['Probability'],
            marker_color=['red', 'blue']  # Color for each bar
        )
    ])
    st.plotly_chart(fig, use_container_width=False)