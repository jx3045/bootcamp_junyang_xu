# app.py
from flask import Flask, request, jsonify
import pickle
import matplotlib.pyplot as plt
import io, base64

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")
    if features is None:
        return jsonify({"error": "No features provided"}), 400
    try:
        prediction = model.predict([features]).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/<float:input1>", methods=["GET"])
def predict_one(input1):
    try:
        prediction = model.predict([[input1]]).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/<float:input1>/<float:input2>", methods=["GET"])
def predict_two(input1, input2):
    try:
        prediction = model.predict([[input1, input2]]).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/plot", methods=["GET"])
def plot():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])  # demo plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_bytes}" />'

if __name__ == "__main__":
    app.run(port=5000, debug=True)
