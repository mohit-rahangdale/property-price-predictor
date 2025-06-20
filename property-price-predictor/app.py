import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('models/house_price_model.pkl')

model = load_model()

@st.cache_data
def load_metadata():
    df = pd.read_csv('data/Bengaluru_House_Data.csv')
    
    # Preprocessing
    df = df.dropna(subset=['size', 'bath', 'price'])
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    df['bhk'] = df['size'].str.split().str.get(0).astype(float)
    
    # area types and locations
    area_types = df['area_type'].unique()
    top_locations = df['location'].value_counts().head(20).index
    return area_types, top_locations

area_types, top_locations = load_metadata()

# Streamlit Ui
st.title('🏠 Bengaluru House Price Predictor')
st.write("Predict property prices in Bengaluru using ML")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        area_type = st.selectbox('Area Type', area_types)
        location = st.selectbox('Location', top_locations)
        total_sqft = st.number_input('Total Sq. Ft.', min_value=300, max_value=10000, value=1000)
        
    with col2:
        bath = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
        balcony = st.number_input('Balconies', min_value=0, max_value=5, value=1)
        bhk = st.number_input('BHK', min_value=1, max_value=10, value=2)
    
    submit_button = st.form_submit_button("Predict Price")

def preprocess_input(area_type, location, total_sqft, bath, balcony, bhk):
    le = LabelEncoder()
    le.fit(area_types)
    area_type_encoded = le.transform([area_type])[0]
    
    le.fit(top_locations)
    location_encoded = le.transform([location])[0] if location in top_locations else 0
    
    return pd.DataFrame([[area_type_encoded, location_encoded, total_sqft, bath, balcony, bhk]],
                        columns=['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk'])

# Prediction
if submit_button:
    input_df = preprocess_input(area_type, location, total_sqft, bath, balcony, bhk)
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Predicted Price: ₹{prediction:,.2f} Lakhs")
    
    st.subheader("Feature Impact")
    feature_names = ['Area Type', 'Location', 'Sq. Ft.', 'Bathrooms', 'Balconies', 'BHK']
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    st.bar_chart(importance_df.set_index('Feature'))

#visualizations
st.markdown("---")
st.subheader("Market Insights")

@st.cache_data
def load_sample_data():
    df = pd.read_csv('data/Bengaluru_House_Data.csv', nrows=500)
    df = df.dropna(subset=['price'])
    return df

sample_df = load_sample_data()

if not sample_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Distribution (₹ Lakhs)**")
        st.bar_chart(sample_df['price'].value_counts().head(10))
    
    with col2:
        st.write("**Avg Price by BHK**")
        avg_price_bhk = sample_df.groupby('size')['price'].mean().reset_index()
        st.line_chart(avg_price_bhk.set_index('size'))