from flask import Flask, request, render_template, jsonify
import re
import pandas as pd
import joblib

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the trained model and vectorizers
model = joblib.load("D:/ThesisCode/Models/randomForest.pkl")
count_vec = joblib.load("D:/ThesisCode/Models/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-classification', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data["text"]

    text_df = pd.DataFrame({"text": [text]})
    text_df["lower_case"] = text_df["text"].apply(
        lambda x: x.lower().strip().replace("\n", " ").replace("\r", " ")
    )
    text_df["alphabetic"] = (
        text_df["lower_case"]
        .apply(lambda x: re.sub(r"[^a-zA-Z\']", " ", x))
        .apply(lambda x: re.sub(r"[^\x00-\x7F]+", "", x))
    )

    text = count_vec.transform([text_df["text"][0]])
    prediction = model.predict(text)
    prediction_list = prediction.tolist()
    response_dict = {"category": prediction_list}
    return jsonify(response_dict)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)