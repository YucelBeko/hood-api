from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Model ve Scaler yükleniyor
model = joblib.load("hood_FDE_Predictor.pkl")

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
    data = request.json.get("features", None)
    if data is None:
        return jsonify({"error": "Veri formatı hatalı. 'features' listesi gerekiyor."}), 400

    try:
        X = np.array(data).reshape(1, -1)

        if USE_SCALER:
            X = scaler.transform(X)

        prediction = model.predict(X)[0]

        return jsonify({"prediction": round(float(prediction), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
