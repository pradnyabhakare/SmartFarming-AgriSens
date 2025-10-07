import pandas as pd
import pickle

# ---------- Load all encoders once ----------
with open('all_crops_encoders.pkl', 'rb') as f:
    all_encoders = pickle.load(f)

# ---------- Map each commodity to its model ----------
commodity_model_files = {
    "Tomato": "student_tomato_model.pkl",
    "Wheat": "wheat_model.pkl",
    "Rice": "rice_model.pkl",
    "Onion": "student_onion_model.pkl",
    "Potato":"student_potato_model.pkl"
    # Add more commodities here
}

# ---------- Load all models at once ----------
models = {}
for commodity, file in commodity_model_files.items():
    try:
        with open(file, 'rb') as f:
            models[commodity] = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Model file not found for {commodity}")

# ---------- Prediction Function ----------
def predict_crop_price(user_input: dict) -> float:
    """
    Predict crop price for given user input.
    user_input must include 'Commodity' key.
    """
    commodity = user_input.get("Commodity")
    if commodity not in models:
        raise ValueError(f"No model found for commodity '{commodity}'")

    model = models[commodity]
    encoders = all_encoders

    # Prepare dataframe
    df = pd.DataFrame([user_input])

    # Encode categorical features
    categorical_cols = ['STATE', 'District Name', 'Market Name', 'Variety', 'Grade', 'Season']
    for col in categorical_cols:
        le = encoders[col]
        val = df.at[0, col]
        if val in le.classes_:
            df.at[0, col] = le.transform([val])[0]
        else:
            df.at[0, col] = 0  # unseen category

    # Feature order must match training
    features = ['STATE', 'District Name', 'Market Name', 'Variety', 'Grade', 'Season', 'Month']
    X = df[features]

    # Predict
    predicted_price = model.predict(X)[0]
    return predicted_price

