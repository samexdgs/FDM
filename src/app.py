from src.utils import load_lottie_files
import pandas as pd
from src.config import LABEL_2_EXERCISE, LABEL_2_TREATMENT, EXERCISE_DIR
from src.paths import MODEL_DIR, ANIMATION_DIR
from streamlit_lottie import st_lottie
import json
import streamlit as st
import pickle
import sys
sys.path.append("../")


# Load model
with open(MODEL_DIR/"logistic_regression.pkl", 'rb') as file:
    model = pickle.load(file)

st.title("🦶 Foot Drop Management App")
st.markdown("""
This app predicts the recommended exercise type and treatment based on patient details.
Please enter the relevant information below.
""")
st.markdown("---")


# Define options for user input based on dataset distribution
severity_options = ["Mild", "Moderate", "Severe"]
cause_options = [
    "Postural imbalance", "Mild nerve weakness", "Spinal cord injury",
    "Peripheral nerve damage", "Stroke", "Trauma to lower limb",
    "Early-stage neuropathy", "Severe neuropathy"
]
assistive_devices_options = ["None", "Cane", "Walker", "AFO", "Wheelchair"]
pain_intensity_options = ["None", "Mild", "Moderate", "Severe"]
pain_frequency_options = ["None", "Occasional", "Frequent", "Constant"]
pain_type_options = ["None", "Aching", "Burning", "Sharp", "Numbness"]

# User input fields
severity = st.selectbox("Severity Level", severity_options)
cause_of_foot_drop = st.selectbox("Cause of Foot Drop", cause_options)
assistive_device = st.selectbox(
    "Assistive Device Used", assistive_devices_options)
pain_intensity = st.selectbox("Pain Intensity", pain_intensity_options)
pain_frequency = st.selectbox("Pain Frequency", pain_frequency_options)
pain_type = st.selectbox("Pain Type", pain_type_options)

all_info = st.text_area("Additional Symptoms/Triggers (Text Input)",
                        "", placeholder="Enter any extra details here")

# Convert user input to a DataFrame (matching the model's expected format)
input_data = pd.DataFrame([{
    "Severity": severity,
    "Cause of Foot Drop": cause_of_foot_drop,
    "Assistive Devices Used": assistive_device,
    "Pain Intensity": pain_intensity,
    "Pain Frequency": pain_frequency,
    "Pain Type": pain_type,
    "All Info": all_info
}])


# Predict button
if st.button("Get Recommendedation"):
    prediction = LABEL_2_EXERCISE.get(model.predict(
        input_data)[0])  # Predict the exercise class
    recommended_treatment = ". ".join(LABEL_2_TREATMENT.get(prediction))

    # Display result with proper formatting
    st.success(f"**Recommended Exercise:** {prediction}")
    st.markdown(f"**Recommended Treatment:**\n{recommended_treatment}")

    # Display Animation

    animations = load_lottie_files(prediction)

    st.title(f":blue[Practise These {len(animations)} Exercises]")

    if animations:
        for i, anim in enumerate(animations):
            st.markdown("---")
            st.markdown(f"**Exercise {i+1}**")
            st_lottie(anim, speed=1, reverse=False, loop=True,
                      quality="high", height=400, width=400)
    else:
        st.warning("No animations found for this exercise.")


# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Samuel")
