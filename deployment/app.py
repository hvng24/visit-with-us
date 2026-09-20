import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="wide")

MODEL_REPO = "hvng24/tourism-model"
MODEL_FILENAME = "best_tourism_model_v1.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    return joblib.load(model_path)


model = load_model()

st.title("Wellness Tourism Package Prediction")
st.markdown(
    "Predict whether a customer is likely to purchase the **Wellness Tourism Package** "
    "based on their profile and interaction history with the sales team."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=25000.0, step=1000.0)

with col2:
    st.subheader("Interaction & Preferences")
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0)
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0.0, max_value=20.0, value=3.0)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns Car?", ["Yes", "No"])
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0)

type_of_contact_map = {"Company Invited": 0, "Self Enquiry": 1}
occupation_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map = {"Female": 0, "Male": 1}
product_pitched_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_status_map = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
designation_map = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact_map[type_of_contact],
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation_map[occupation],
    "Gender": gender_map[gender],
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched_map[product_pitched],
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status_map[marital_status],
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation_map[designation],
    "MonthlyIncome": monthly_income,
}])

st.markdown("---")
if st.button("Predict Purchase Likelihood", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package (confidence: {probability * 100:.1f}%)")
    else:
        st.warning(f"Unlikely to purchase the Wellness Tourism Package (confidence: {(1 - probability) * 100:.1f}%)")

    st.metric("Predicted Probability of Purchase", f"{probability * 100:.1f}%")
