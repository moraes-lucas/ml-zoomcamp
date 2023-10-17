import pickle

def load_bin(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

dv = load_bin('dv.bin')
model = load_bin('model1.bin')

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0][1]

print(y_pred)