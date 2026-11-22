import pickle
from flask import Flask, request, jsonify

model_file = 'rf_model.bin'

with open(model_file, 'rb') as f_in:
    dv, le, model = pickle.load(f_in)

app = Flask('sleep')


@app.route('/predict', methods=['POST'])
def predict():
    patient_data = request.get_json()

    X = dv.transform([patient_data])
    pred_idx = model.predict(X)[0]
    prediction = le.inverse_transform([pred_idx])[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)