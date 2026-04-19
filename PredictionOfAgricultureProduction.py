from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
le_crop = pickle.load(open("crop.pkl", "rb"))
le_state = pickle.load(open("state.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    crop = request.form['crop']
    state = request.form['state']
    a2fl = float(request.form['a2fl'])
    c2 = float(request.form['c2'])
    pc = float(request.form['pc'])

    crop_encoded = le_crop.transform([crop])[0]
    state_encoded = le_state.transform([state])[0]

    features = np.array([[crop_encoded, state_encoded, a2fl, c2, pc]])

    prediction = model.predict(features)[0]

    return render_template("index.html", result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
