import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./datasetV5.csv")

print(data.head())

X = data[["FIRST_CONTEXT", "LAST_CONTEXT"]]
y = data["GENRE"]

test_size_ratio = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_ratio, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(
    X_train.apply(lambda row: f"{row['FIRST_CONTEXT']} {row['LAST_CONTEXT']}", axis=1)
)
X_test_tfidf = vectorizer.transform(
    X_test.apply(lambda row: f"{row['FIRST_CONTEXT']} {row['LAST_CONTEXT']}", axis=1)
)

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred, zero_division=0
)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(report)

def predict_genre(first_context, last_context):
    context_combined = f"{first_context} {last_context}"
    context_tfidf = vectorizer.transform([context_combined])
    predicted_genre = classifier.predict(context_tfidf)
    return predicted_genre[0]


first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(first_context, last_context)
print(f"Predicted Genre: {predicted_genre}")


