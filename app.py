import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import spacy
import joblib

nlp = spacy.load("en_core_web_sm")

# ---------------- Soil Map (Hindi + English) ----------------
soil_map = {
    "काली मिट्टी (Black Soil)": {"N": 80, "P": 45, "K": 50, "ph": 6.5, "temperature": 25, "humidity": 60, "rainfall": 200},
    "गीली मिट्टी (Wet Soil)": {"N": 60, "P": 40, "K": 45, "ph": 6.0, "temperature": 22, "humidity": 75, "rainfall": 250},
    "लाल मिट्टी (Red Soil)": {"N": 40, "P": 30, "K": 35, "ph": 5.8, "temperature": 28, "humidity": 55, "rainfall": 180},
    "रेतीली मिट्टी (Sandy Soil)": {"N": 30, "P": 20, "K": 25, "ph": 6.2, "temperature": 30, "humidity": 40, "rainfall": 150},
    "दोमट मिट्टी (Loamy Soil)": {"N": 70, "P": 40, "K": 40, "ph": 6.8, "temperature": 26, "humidity": 65, "rainfall": 220},
    "पीली मिट्टी (Yellow Soil)": {"N": 35, "P": 25, "K": 30, "ph": 6.0, "temperature": 27, "humidity": 50, "rainfall": 160},
    "चिकनी मिट्टी (Clay Soil)": {"N": 75, "P": 50, "K": 55, "ph": 6.7, "temperature": 24, "humidity": 70, "rainfall": 230},
    "अल्कलाइन मिट्टी (Alkaline Soil)": {"N": 25, "P": 15, "K": 20, "ph": 8.2, "temperature": 29, "humidity": 35, "rainfall": 100},
    "अम्लीय मिट्टी (Acidic Soil)": {"N": 45, "P": 35, "K": 30, "ph": 5.0, "temperature": 23, "humidity": 80, "rainfall": 300},
    "जलोढ़ मिट्टी (Alluvial Soil)": {"N": 65, "P": 40, "K": 45, "ph": 6.4, "temperature": 25, "humidity": 68, "rainfall": 210},
}

# ---------------- Season Map ----------------
season_map = {
    "रबी (Rabi)": "Rabi",
    "खरीफ (Kharif)": "Kharif",
    "जायद (Zaid)": "Zaid"
}

# ---------------- ZIP Loader ----------------
def load_csv_from_zip(zip_path, extract_to="dataset"):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    csv_files = [f for f in os.listdir(extract_to) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError("No CSV file found in ZIP!")

    return os.path.join(extract_to, csv_files[0])

# ---------------- Train MODEL WITH ACCURACY ----------------
def train_crop_model(zip_path):
    csv_path = load_csv_from_zip(zip_path)
    df = pd.read_csv(csv_path)

    rename_map = {
        "Nitrogen": "N", "Phosphorus": "P", "Potassium": "K",
        "Temperature": "temperature", "Humidity": "humidity",
        "PH": "ph", "pH": "ph", "Rainfall": "rainfall"
    }
    df.rename(columns=rename_map, inplace=True)

    le_season = None
    if "Season" in df.columns:
        le_season = LabelEncoder()
        df["Season"] = le_season.fit_transform(df["Season"])

    target_col = "label" if "label" in df.columns else "crop"

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train–Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n✅ Model Training Complete")
    print(f"📊 Training Accuracy: {acc * 100:.2f}%")

    # Save model + accuracy
    joblib.dump(model, "crop_predictor_model.pkl")
    joblib.dump(le, "crop_label_encoder.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")
    joblib.dump(acc, "model_accuracy.pkl")

    if le_season:
        joblib.dump(le_season, "season_label_encoder.pkl")

    return model, le, list(X.columns), le_season, acc

# ---------------- Prediction ----------------
def predict_top_crops(model, le, feature_names, params, top_k=5):
    df_input = pd.DataFrame([{col: params.get(col, 0) for col in feature_names}])
    probs = model.predict_proba(df_input)[0]
    probs = probs / probs.sum()

    idx = np.argsort(probs)[::-1][:top_k]

    crops = le.inverse_transform(idx)
    scores = np.round(probs[idx] * 100, 2)

    return list(crops), list(scores)

# ---------------- MAIN FUNCTION ----------------
def main(zip_path, params, top_k=5):

    model_exists = os.path.exists("crop_predictor_model.pkl")
    accuracy_exists = os.path.exists("model_accuracy.pkl")

    # -------- LOAD IF ALL FILES EXIST --------
    if model_exists and accuracy_exists:
        model = joblib.load("crop_predictor_model.pkl")
        le = joblib.load("crop_label_encoder.pkl")
        feature_names = joblib.load("feature_names.pkl")
        acc = joblib.load("model_accuracy.pkl")

        print(f"\n📊 Loaded Model Accuracy: {acc * 100:.2f}%")

        le_season = (
            joblib.load("season_label_encoder.pkl")
            if os.path.exists("season_label_encoder.pkl")
            else None
        )

    # -------- TRAIN IF ANY FILE IS MISSING --------
    else:
        print("⚠️ Model or Accuracy file missing. Training again...")
        model, le, feature_names, le_season, acc = train_crop_model(zip_path)
        print(f"🎯 Saved Accuracy = {acc * 100:.2f}%")

    # -------- PREDICT --------
    crops, scores = predict_top_crops(model, le, feature_names, params, top_k)

    print("\n🌾 Top Recommended Crops:")
    for crop, score in zip(crops, scores):
        print(f"✔ {crop} — {score}%")

    return crops, scores, acc


# ---------------- RUN ----------------
if __name__ == "__main__":
    zip_path = "C:/Users/singh/Downloads/Crop_recommendation 1.zip"

    params = {
        "N": 60,
        "P": 40,
        "K": 45,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 200
    }

    main(zip_path, params, top_k=5)
