import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder as le

model_file = 'rf_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('sleep')

@app.route('/predict', methods=['POST'])
def predict_dump():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    sleep = y_pred >= 0.5
    
    result = {
        'sleep_probability': float(y_pred),
        'sleep': bool(sleep)
    }
    return jsonify(result)

def predict():
    patient_data = request.get_json()
    # patient_data = json.loads(incoming_json)
    X = dv.transform([patient_data])
    pred_idx = model.predict(X)[0]
    prediction = le.inverse_transform([pred_idx])[0]

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) # localhost:9696/ping
