import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_excel('./datasetV8.xlsx')



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

data['TEXT'] = data['FIRST_CONTEXT'] + ' ' + data['LAST_CONTEXT']

X = data['TEXT']
y = data['GENRE']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_classifier = SVC(C=1.0, kernel='poly', degree=3, gamma='scale', coef0=1.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1, decision_function_shape='ovr')
svm_classifier.fit(X_train_tfidf, y_train)

y_pred = svm_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

def predict_genre(context):
    context_tfidf = vectorizer.transform([context])
    predicted_genre_idx = svm_classifier.predict(context_tfidf)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_idx])[0]
    return predicted_genre

first_context = "diet leads to name change one of the enemies of the right wing republic of american politics michael moore has changed his name to michael less following a recent weight loss programme moore who challenged american gun laws as well as a number of other important american beliefs in his time lost nearly 6 stone on a diet of "
last_context = "on the count of three one two three bzzzzztttt each team has also been given a list of store numbers and addresses the contents of each truck must be inventoried this is your inventory sheet she waved a pink preprinted pad over her head it doesnt really matter what we send to these stores only that we send them a variety of merchandise if you open a truck thats filled with one item then start unloading share the contents with your neighbors then take the yellow pad she began to wave that in  "
predicted_genre = predict_genre(f'{first_context} {last_context}')
print(f'Predicted Genre: {predicted_genre}')


# save the model
joblib.dump(svm_classifier, 'svm_model.pkl')
joblib.dump(vectorizer, 'svm_vectorizer.pkl')
joblib.dump(label_encoder, 'svm_label_encoder.pkl')

# load the model
svm_model = joblib.load('svm_model.pkl')
svm_vectorizer = joblib.load('svm_vectorizer.pkl')
svm_label_encoder = joblib.load('svm_label_encoder.pkl')

# input the text to predict
context = input("Enter the text to predict: ")
print(f"Predicted Genre: {predict_genre(context)}")
