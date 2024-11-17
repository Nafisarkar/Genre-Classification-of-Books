import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_excel('./datasetV6.xlsx')

# Display the first few rows of the dataframe
print(df.head())


# Ensure that the target columns are binary indicators
df[['COMEDY', 'FICTION', 'MYSTERY', 'ROMANCE', 'HORROR', 'THRILLER', 'CRIME', 'PLAY']] = df[
    ['COMEDY', 'FICTION', 'MYSTERY', 'ROMANCE', 'HORROR', 'THRILLER', 'CRIME', 'PLAY']
].astype(int)

# Concatenate 'FIRST_CONTEXT', 'LAST_CONTEXT', and 'AUTHOR' into a single feature column
df['FEATURES'] = df['FIRST_CONTEXT'] + ' ' + df['LAST_CONTEXT'] + ' ' + df['AUTHOR'].apply(lambda x: ' '.join(map(str, x)))

# Select relevant columns for training
features = ['FEATURES']
labels = df[['COMEDY', 'FICTION', 'MYSTERY', 'ROMANCE', 'HORROR', 'THRILLER', 'CRIME', 'PLAY']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Print classification report
print(classification_report(y_test, y_pred))