import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load_bin(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load_bin('dv.bin')
model = load_bin('model1.bin')

app = Flask('bank-credict-scoring')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0][1]

    result = {
        'get_credit_probability': float(y_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)