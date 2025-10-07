from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import pickle
from user import predict_crop_price
from utils import recommend_crops

app = Flask(__name__)

# -------------------------------
# Load dataset for dropdowns-price_prediction
# -------------------------------
df = pd.read_csv("processed_crop_prices.csv")

# -------------------------------
# Load dataset for crop_recommendation
# -------------------------------
base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, "crop_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(base_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
with open(os.path.join(base_path, "feature_names.pkl"), "rb") as f:
    features = pickle.load(f)

@app.route('/')
def home():
    return render_template("landingpage.html")

@app.route('/price')
def index():
    # Initial dropdown data
    states = sorted(df["STATE"].dropna().unique())
    commodities = sorted(df["Commodity"].dropna().unique())
    seasons = sorted(df["Season"].dropna().unique())
    return render_template("price.html", states=states, commodities=commodities, seasons=seasons)

@app.route('/disease')
def index1():
    return render_template("disease.html")

@app.route('/smart')
def index2():
    return render_template("smart.html")

# -------------------------------
# Dynamic Dropdown Routes- price_prediction
# -------------------------------
@app.route('/get_districts', methods=['POST'])
def get_districts():
    state = request.form['state']
    districts = sorted(df[df["STATE"] == state]["District Name"].dropna().unique())
    return jsonify(districts)

@app.route('/get_markets', methods=['POST'])
def get_markets():
    state = request.form['state']
    district = request.form['district']
    markets = sorted(df[(df["STATE"] == state) & (df["District Name"] == district)]["Market Name"].dropna().unique())
    return jsonify(markets)

@app.route('/get_varieties_grades', methods=['POST'])
def get_varieties_grades():
    commodity = request.form['commodity']
    varieties = sorted(df[df["Commodity"] == commodity]["Variety"].dropna().unique())
    grades = sorted(df[df["Commodity"] == commodity]["Grade"].dropna().unique())
    return jsonify({"varieties": varieties, "grades": grades})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            "Commodity": request.form["Commodity"],
            "STATE": request.form["STATE"],
            "District Name": request.form["District Name"],
            "Market Name": request.form["Market Name"],
            "Variety": request.form["Variety"],
            "Grade": request.form["Grade"],
            "Season": request.form["Season"],
            "Month": int(request.form["Month"])
        }

        price = predict_crop_price(user_input)

        return jsonify({"price": round(float(price), 2)})

    except Exception as e:
        return jsonify({"error": str(e)})    

#****************************************************************

#-------------------------------
# Crop Recommendation Route
#-------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Extract input
        N = float(request.form.get("N", 0))
        P = float(request.form.get("P", 0))
        K = float(request.form.get("K", 0))
        temperature = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))
        ph = float(request.form.get("ph", 0))
        rainfall = float(request.form.get("rainfall", 0))

        print("📩 Received values:", N, P, K, temperature, humidity, ph, rainfall)

        # Prepare sample dataframe
        sample = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        # ✅ Get raw results (no filtering)
        results_raw = recommend_crops(model, sample, label_encoder, features)

        results_clean = {
            category: [(crop, float(prob)) for crop, prob in items]
            for category, items in results_raw.items()
        }

        print("✅ Recommendation successful:", results_clean)

        # -------------------------------
        # Return JSON to frontend
        # -------------------------------
        return jsonify(results_clean)

    except Exception as e:
        print("❌ ERROR in /recommend:", e)
        return jsonify({"error": str(e)})

#****************************************************************

if __name__ == '__main__':
    app.run(debug=True)