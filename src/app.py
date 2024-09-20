from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo
model = joblib.load('fraud_detection_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "API de Detecção de Fraudes está ativa!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)
        result = {'prediction': int(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
