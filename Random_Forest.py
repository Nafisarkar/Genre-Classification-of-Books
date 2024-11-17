import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_excel("./datasetV8.xlsx")


for col in data.columns:
    data[col] = data[col].apply(lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))

for col in data.columns:
    data[col] = data[col].apply(lambda x: re.sub(r'[^a-zA-Z\']', ' ', x.lower()))
    data[col] = data[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    data[col] = data[col].apply(lambda x: re.sub(r'http\S+', '', x))

data['TEXT_LENGTH'] = data['FIRST_CONTEXT'].apply(len)

fig = plt.figure(figsize=(14,7))
data['length'] = data['TEXT_LENGTH']
ax1 = fig.add_subplot(122)
sns.histplot(data['length'], ax=ax1, color='green')
describe = data.length.describe().to_frame().round(2)
ax2 = fig.add_subplot(121)
ax2.axis('off')
font_size = 14
bbox = [0, 0, 1, 1]
table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)
fig.suptitle('Distribution of text length for text.', fontsize=16)
plt.show()

sns.set_theme(style="whitegrid")
sns.countplot(x=data["GENRE"])
plt.show()



X = data[["FIRST_CONTEXT", "LAST_CONTEXT"]]
y = data["GENRE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(
    X_train.apply(lambda row: f"{row['FIRST_CONTEXT']} {row['LAST_CONTEXT']}", axis=1)
)
X_test_tfidf = vectorizer.transform(
    X_test.apply(lambda row: f"{row['FIRST_CONTEXT']} {row['LAST_CONTEXT']}", axis=1)
)

classifier = RandomForestClassifier(n_estimators=200, max_depth=27, random_state=42)
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


def predict_genre(context):
    context_tfidf = vectorizer.transform([context])
    predicted_genre = classifier.predict(context_tfidf)
    return predicted_genre[0]


context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of " + "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(context)
print(f"Predicted Genre: {predicted_genre}")

from sklearn.preprocessing import StandardScaler
transformer = StandardScaler()
import joblib

# save the model
joblib.dump(classifier, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(transformer, 'transformer.pkl')


# input the text to predict
context = input("Enter the text to predict: ")
print(f"Predicted Genre: {predicted_genre[0]}")
