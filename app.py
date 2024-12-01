import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
import joblib

# Load the trained model, scaler, and feature selector
ann_model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('feature_selector.pkl')

# Load the breast cancer dataset for feature names
data = load_breast_cancer()
all_feature_names = list(data.feature_names)  # All feature names
selected_features_indices = selector.get_support(indices=True)  # Get indices of selected features
selected_feature_names = [all_feature_names[i] for i in selected_features_indices]  # Get selected feature names

# Get feature range from the scaler (for creating sliders)
scaled_min = scaler.data_min_
scaled_max = scaler.data_max_

# Streamlit application
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a breast tumor is malignant or benign based on clinical characteristics.")

# User input for selected patient characteristics (only selected features)
st.header("Enter Patient Characteristics")

# Create sliders for each selected feature
features = {}
for feature_name in selected_feature_names:
    i = all_feature_names.index(feature_name)  # Find index of selected feature in all features
    min_val = float(scaled_min[i])
    max_val = float(scaled_max[i])
    default_val = (min_val + max_val) / 2
    features[feature_name] = st.slider(
        feature_name,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val - min_val) / 100
    )

# Prediction button
if st.button("Predict"):
    # Create an array for all 30 features, with default values for non-selected features
    full_input_data = np.zeros((1, len(all_feature_names)))
    
    # Fill in the selected features with user inputs
    for i, feature_name in enumerate(all_feature_names):
        if feature_name in selected_feature_names:
            feature_index = selected_feature_names.index(feature_name)
            full_input_data[0, i] = features[feature_name]
        else:
            # Set non-selected features to the median of their scaled range
            full_input_data[0, i] = (scaled_min[i] + scaled_max[i]) / 2  # Default value

    # Scale the full input data (30 features)
    input_data_scaled = scaler.transform(full_input_data)
    
    # Extract only the selected features (the model was trained on these 10 features)
    input_data_scaled_selected = input_data_scaled[:, selector.get_support(indices=True)]
    
    # Get prediction (using only the 10 selected features)
    prediction = ann_model.predict(input_data_scaled_selected)
    
    # Display result
    if prediction[0] == 1:
        st.success("The model predicts: **Benign (1)**")
    else:
        st.error("The model predicts: **Malignant (0)**")

# Additional description
st.write("### Model Description")
st.write("This model predicts whether a tumor is malignant or benign based on clinical characteristics. It uses a trained Artificial Neural Network (ANN) model.")
