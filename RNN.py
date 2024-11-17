import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('./datasetV5.csv')

data['TEXT'] = data['FIRST_CONTEXT'] + ' ' + data['LAST_CONTEXT']

X = data['TEXT']
y = data['GENRE']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_sequence_length = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Build RNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
def predict_genre_rnn(first_context, last_context):
    context_combined = f"{first_context} {last_context}"
    context_seq = tokenizer.texts_to_sequences([context_combined])
    context_pad = pad_sequences(context_seq, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(context_pad)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)
    predicted_genre = label_encoder.inverse_transform([predicted_class[0]])[0]
    return predicted_genre

import numpy as np

first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre_rnn(first_context, last_context)
print(f'Predicted Genre: {predicted_genre}')