from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Model, Scaler ve Feature Sırası Yükleniyor
model = joblib.load("model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
    USE_SCALER = True
except:
    scaler = None
    USE_SCALER = False

try:
    feature_names = joblib.load("feature.pkl")  # list of encoded feature names
except:
    feature_names = None
    raise ValueError("feature.pkl bulunamadı. Gerekli sütun isimleri olmadan tahmin yapılamaz.")

@app.route('/')
def home():
    return "✅ Hood FDE Predictor API aktif."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("features", None)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Veri formatı hatalı. 'features' sözlüğü gerekiyor."}), 400

    try:
        # 🔄 Dict → DataFrame
        df_input = pd.DataFrame([data])  # tek satır
        df_input_encoded = pd.get_dummies(df_input)

        # 🧱 Eksik sütunları tamamla
        for col in feature_names:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[feature_names]  # doğru sıraya sok

        # 🔄 Standardizasyon (varsa)
        X = df_input_encoded.values
        if USE_SCALER:
            X = scaler.transform(X)

        # 🔍 Tahmin
        prediction = model.predict(X)[0]

        return jsonify({"prediction": round(float(prediction), 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
