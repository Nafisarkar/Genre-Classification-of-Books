import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./datasetV5.csv")

data["TEXT"] = data["FIRST_CONTEXT"] + " " + data["LAST_CONTEXT"]

X = data["TEXT"]
y = data["GENRE"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#10 = 36
#100 = 41
#1000 = 60
#3000 = 41

max_length = 1000
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=128, input_length=max_length),
        Conv1D(filters=128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units=128, activation="relu"),
        Dense(units=len(label_encoder.classes_), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

#10 = 36
#100 = 38
#1000 = 25

epochsSize = 10 # set epochSize here
history = model.fit(
    X_train_pad,
    y_train,
    epochs=epochsSize,
    batch_size=32,
    validation_data=(X_test_pad, y_test),
)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy}")

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


def predict_genre(first_context, last_context):
    context_combined = f"{first_context} {last_context}"
    context_seq = tokenizer.texts_to_sequences([context_combined])
    context_pad = pad_sequences(context_seq, maxlen=max_length)
    predicted_probabilities = model.predict(context_pad)[0]
    predicted_genre_idx = np.argmax(predicted_probabilities)
    predicted_genre = label_encoder.inverse_transform([predicted_genre_idx])[0]
    return predicted_genre


first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(first_context, last_context)
print(f"Predicted Genre: {predicted_genre}")

print(f"accuracy{accuracy} with epoch size {epochsSize} max length {max_length}")
