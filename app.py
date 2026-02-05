# app.py - Streamlit Application for House Price Prediction

import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_house_price_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save and load the scaler for scaling user input

# Define function to make prediction
def predict_house_price(input_data):
    # Scale input data
    scaled_data = scaler.transform([input_data])
    
    # Predict price
    prediction = model.predict(scaled_data)
    return prediction[0]

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the details of the house:")

# User input
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=1500)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=1000, max_value=20000, value=5000)
floors = st.number_input("Number of Floors", min_value=1, max_value=3, value=2)
waterfront = st.selectbox("Waterfront (0: No, 1: Yes)", [0, 1])
view = st.number_input("View Quality (0-4)", min_value=0, max_value=4, value=0)
condition = st.number_input("Condition (1-5)", min_value=1, max_value=5, value=3)
sqft_above = st.number_input("Area Above Ground (sqft)", min_value=500, max_value=10000, value=1000)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=0)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2025, value=0)

# Prepare input for prediction
input_data = np.array([bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                       condition, sqft_above, sqft_basement, yr_built, yr_renovated])

# Display prediction
if st.button("Predict House Price"):
    prediction = predict_house_price(input_data)
    st.write(f"Predicted House Price: ${prediction:,.2f}")
