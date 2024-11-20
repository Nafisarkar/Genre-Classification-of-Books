from flask import Flask, request, render_template, jsonify
import re
import pandas as pd
import joblib

# add pdf support
# image support
# model for 3 - Done

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/text-classification", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data["text"]

    model = data["model"]
    if model == "random_forest":
        model = joblib.load("Models/randomForest.pkl")
        count_vec = joblib.load("Models/randomvectorizer.pkl")
        text_df = pd.DataFrame({"text": [text]})
        text_df["lower_case"] = text_df["text"].apply(lambda x: x.lower().strip().replace("\n", " ").replace("\r", " "))
        text_df["alphabetic"] = (
            text_df["lower_case"]
            .apply(lambda x: re.sub(r"[^a-zA-Z\']", " ", x))
            .apply(lambda x: re.sub(r"[^\x00-\x7F]+", "", x))
        )

        text_transformed = count_vec.transform([text_df["alphabetic"][0]])
        if text_transformed.shape[0] == 0:
            return (
                jsonify({"error": "No valid features extracted from the input text"}),
                400,
            )
        prediction = model.predict(text_transformed)
        prediction_list = prediction.tolist()
        response_dict = {"category": prediction_list}

    elif model == "svm":
        model = joblib.load("Models/svm_model.pkl")
        vectorizer = joblib.load("Models/svm_vectorizer.pkl")
        label_encoder = joblib.load("Models/svm_label_encoder.pkl")

        text_df = pd.DataFrame({"text": [text]})
        text_df["lower_case"] = text_df["text"].apply(lambda x: x.lower().strip().replace("\n", " ").replace("\r", " "))
        text_df["alphabetic"] = (
            text_df["lower_case"]
            .apply(lambda x: re.sub(r"[^a-zA-Z\']", " ", x))
            .apply(lambda x: re.sub(r"[^\x00-\x7F]+", "", x))
        )
        text_transformed = vectorizer.transform([text_df["alphabetic"][0]])
        prediction = model.predict(text_transformed)
        prediction_list = prediction.tolist()
        predicted_genre = label_encoder.inverse_transform(prediction_list)[0]
        response_dict = {"category": predicted_genre}

    elif model == "gbm":
        model = joblib.load("Models/gbm_model.pkl")
        vectorizer = joblib.load("Models/gbm_vectorizer.pkl")
        label_encoder = joblib.load("Models/gbm_label_encoder.pkl")

        text_df = pd.DataFrame({"text": [text]})
        text_df["lower_case"] = text_df["text"].apply(lambda x: x.lower().strip().replace("\n", " ").replace("\r", " "))
        text_df["alphabetic"] = (
            text_df["lower_case"]
            .apply(lambda x: re.sub(r"[^a-zA-Z\']", " ", x))
            .apply(lambda x: re.sub(r"[^\x00-\x7F]+", "", x))
        )
        text_transformed = vectorizer.transform([text_df["alphabetic"][0]])
        prediction = model.predict(text_transformed)
        prediction_list = prediction.tolist()
        predicted_genre = label_encoder.inverse_transform(prediction_list)[0]
        response_dict = {"category": predicted_genre}

    else:
        return jsonify({"error": "Invalid model selected"}), 400

    return jsonify(response_dict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

