import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('./datasetV5.csv')

data['TEXT'] = data['FIRST_CONTEXT'] + ' ' + data['LAST_CONTEXT']

X = data['TEXT']
y = data['GENRE']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

gbm_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gbm_classifier.fit(X_train_tfidf, y_train)

y_pred = gbm_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

def predict_genre(first_context, last_context):
    context_combined = f"{first_context} {last_context}"
    context_tfidf = vectorizer.transform([context_combined])
    predicted_genre_idx = gbm_classifier.predict(context_tfidf)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_idx])[0]
    return predicted_genre

first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(first_context, last_context)
print(f'Predicted Genre: {predicted_genre}')

