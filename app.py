from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Model, scaler, ve sütun sırasını yükle
model = joblib.load("model.pkl")
expected_columns = joblib.load("expected_columns.pkl")  # 🔥 burası yeni
try:
    scaler = joblib.load("scaler.pkl")
    USE_SCALER = True
except:
    scaler = None
    USE_SCALER = False

@app.route('/')
def home():
    return "✅ Hood FDE Predictor API aktif."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        incoming_data = request.json.get("features", None)
        if incoming_data is None:
            return jsonify({"error": "Veri formatı hatalı. 'features' dictionary gerekiyor."}), 400

        # 🔄 Dict → DataFrame
        df_input = pd.DataFrame([incoming_data])

        # 🧠 One-hot encoding
        df_encoded = pd.get_dummies(df_input)

        # ⛑️ Eksik sütunları 0 ile doldur
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # ✅ Sıralama
        df_encoded = df_encoded[expected_columns]

        # 🧪 Scaler varsa uygula
        X = df_encoded.values
        if USE_SCALER:
            X = scaler.transform(X)

        # 🔮 Tahmin
        prediction = model.predict(X)[0]
        return jsonify({"prediction": round(float(prediction), 4)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
