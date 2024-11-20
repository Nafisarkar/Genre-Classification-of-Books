import joblib
import pandas as pd
import re

def predict_genre(text):
    model = joblib.load("Models/svm_model.pkl")
    vectorizer = joblib.load("Models/svm_vectorizer.pkl")
    label_encoder = joblib.load("Models/svm_label_encoder.pkl")

    text_df = pd.DataFrame({"text": [text]})
    text_df["lower_case"] = text_df["text"].apply(
        lambda x: x.lower().strip().replace("\n", " ").replace("\r", " ")
    )
    text_df["alphabetic"] = (
        text_df["lower_case"]
        .apply(lambda x: re.sub(r"[^a-zA-Z\']", " ", x))
        .apply(lambda x: re.sub(r"[^\x00-\x7F]+", "", x))
    )

    text_transformed = vectorizer.transform([text_df["alphabetic"][0]])
    if text_transformed.shape[0] == 0:
        return "No valid features extracted from the input text"
    prediction = model.predict(text_transformed)
    prediction_list = prediction.tolist()
    predicted_genre = label_encoder.inverse_transform(prediction_list)[0]
    return predicted_genre


if __name__ == "__main__":
    text = input("Enter the text to predict: ")
    predicted_genre = predict_genre(text)
    print(f"Predicted Genre: {predicted_genre}")