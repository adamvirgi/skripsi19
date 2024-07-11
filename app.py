import pickle
import streamlit as st

# Load the trained scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('svm_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('SVM Classifier for Predicting bb/tb')

    # Get input features from the user
    age = st.number_input('Age', min_value=0.0, max_value=100.0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    height = st.number_input('Height (cm)', min_value=0.0, max_value=250.0)
    weight = st.number_input('Weight (kg)', min_value=0.0, max_value=200.0)
    gender = 0 if gender == "Female" else 1
    # Preprocess the input features
    input_data = [[age, gender, height, weight]]
    input_data_scaled = scaler.transform(input_data)

    # Predict the bb/tb using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    st.write('Predicted bb/tb:', prediction[0])

if __name__ == '__main__':
    main()
