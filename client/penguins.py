import streamlit as st
import requests

# Streamlit app title
st.title("Penguin Species Prediction App")

# Streamlit app sidebar title
st.sidebar.title("Feature Inputs")

# Sidebar inputs for feature columns
island_options = ['Biscoe', 'Dream', 'Torgersen']
sex_options = ['Female', 'Male']
bill_length_mm = st.sidebar.slider("Bill Length (mm)", min_value=30.0, max_value=60.0, value=40.0)
bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", min_value=10.0, max_value=25.0, value=15.0)
flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", min_value=150.0, max_value=240.0, value=200.0)
body_mass_g = st.sidebar.slider("Body Mass (g)", min_value=2000.0, max_value=7000.0, value=4000.0)
island = st.sidebar.selectbox("Island", island_options)
sex = st.sidebar.selectbox("Sex", sex_options)

# Streamlit app button to make a prediction
if st.sidebar.button("Make Prediction"):
    # Prepare feature inputs as a dictionary
    features = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island": island,
        "sex": sex
    }

    # Make a POST request to the FastAPI server for prediction
    url = "http://server:8000/predict"
    response = requests.post(url, json = features)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"The predicted species is: {prediction}")
    else:
        st.error("Error making prediction. Please try again.")