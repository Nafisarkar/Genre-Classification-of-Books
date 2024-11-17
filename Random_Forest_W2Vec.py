import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

data = pd.read_csv('./datasetV5.csv')

data['TEXT'] = data['FIRST_CONTEXT'] + ' ' + data['LAST_CONTEXT']

X = data['TEXT']
y = data['GENRE']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_sentences = [simple_preprocess(sentence) for sentence in X_train.values]

word2vec_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.init_sims(replace=True)

def sent2vec(sen, model):
    vec = np.zeros(model.vector_size)
    count = 0
    for word in sen:
        try:
            vec += model.wv[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

X_train_vec = np.array([sent2vec(sen, word2vec_model) for sen in train_sentences])
X_test_vec = np.array([sent2vec(sen, word2vec_model) for sen in X_test.values])

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vec, y_train)

y_pred = rf_classifier.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

def predict_genre(first_context, last_context):
    context_combined = f"{first_context} {last_context}"
    context_tokens = simple_preprocess(context_combined)
    context_vec = sent2vec(context_tokens, word2vec_model)
    predicted_genre_idx = rf_classifier.predict([context_vec])[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_idx])[0]
    return predicted_genre

first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(first_context, last_context)
print(f'Predicted Genre: {predicted_genre}')
