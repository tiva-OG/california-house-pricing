import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

# load the regressor model and standard scaler
regressor = pickle.load(open("regressor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(data.values())
    print(np.array(list(data.values())).reshape(1, -1))
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regressor.predict(scaled_data)
    print(output[0])
    
    return jsonify(output[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(r) for r in request.form.values()]
    model_inp = scaler.transform(np.array(data).reshape(1, -1))
    print(model_inp)
    output = regressor.predict(model_inp)[0]
    
    return render_template("home.html", prediction_text=f"The prediction for the house price ${100000*output:,.3f}")


if __name__ == "__main__":
    app.run(debug=True)