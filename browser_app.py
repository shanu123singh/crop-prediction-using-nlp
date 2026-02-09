import streamlit as st
from io import BytesIO
from gtts import gTTS
from app import train_crop_model, soil_map, season_map, predict_top_crops
import warnings

# Suppress Streamlit missing ScriptRunContext warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ---------------- Text-to-Speech ----------------
def speak_text_in_browser(text):
    tts = gTTS(text, lang="hi")
    audio = BytesIO()
    tts.write_to_fp(audio)
    st.audio(audio.getvalue(), format="audio/mp3")

# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(
        page_title="🌾 Crop Recommendation Chat Bot",
        layout="wide"
    )
    st.title("🌾 Crop Recommendation Chat Bot | फसल सुझाव चैट बॉट")
    st.caption("AI Based Crop Recommender System | AI आधारित फसल सलाह प्रणाली")

    # Load model only once
    if "model" not in st.session_state:
        zip_path = r"C:\Users\singh\Downloads\Crop_recommendation 1.zip"
        with st.spinner("📊 Training Model... | मॉडल ट्रेन हो रहा है..."):
            # Ensure we unpack 5 values
            model, le, feature_names, le_season, acc = train_crop_model(zip_path)

        st.session_state.model = model
        st.session_state.le = le
        st.session_state.feature_names = feature_names
        st.session_state.le_season = le_season
        st.session_state.acc = acc

        st.success(f"✅ Model Ready! | मॉडल तैयार है! Accuracy: {acc*100:.2f}%")

    # ---------------- Soil Input ----------------
    st.subheader("🧱 मिट्टी का प्रकार चुनें | Select Soil Type")
    soil_choice = st.selectbox(
        "मिट्टी चुनें / Choose Soil Type",
        list(soil_map.keys())
    )
    soil_params = soil_map[soil_choice].copy()

    # ---------------- Season Input ----------------
    st.subheader("🌦️ मौसम चुनें | Select Season")
    season_choice = st.selectbox(
        "मौसम चुनें / Choose Season",
        list(season_map.keys())
    )

    season_value = season_map[season_choice]  # English value for LabelEncoder
    if st.session_state.le_season:
        season_value = st.session_state.le_season.transform([season_value])[0]

    soil_params["Season"] = season_value

    # ---------------- Top K Crops ----------------
    st.subheader("🌱 फसल सुझावों की संख्या | Number of Crop Recommendations")
    top_k = st.slider("कितने सुझाव? / How many crops?", 3, 10, 5)

    # ---------------- Predict Button ----------------
    if st.button("📌 Show Recommendations | फसल सुझाव दिखाएँ"):
        crops, scores = predict_top_crops(
            st.session_state.model,
            st.session_state.le,
            st.session_state.feature_names,
            soil_params,
            top_k=top_k
        )

        st.success(f"🌱 Top {top_k} Recommended Crops | शीर्ष {top_k} फसल अनुशंसाएँ:")
        for i, (crop, score) in enumerate(zip(crops, scores), start=1):
            st.write(f"**{i}. {crop}** — Confidence / विश्वसनीयता: {score}%")

        # Speak the recommendation in Hindi
        speak_text_in_browser(f"आपके लिए सर्वोत्तम {top_k} फसल सुझाव हैं: {', '.join(crops)}")

if __name__ == "__main__":
    main()
