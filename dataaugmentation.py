import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from random import choice, randint

# Load the existing data
data = pd.read_excel("./datasetV8.xlsx")

data['GENRE'].value_counts()

# Define a function to perform text augmentation
def augment_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Apply random operations to the tokens
    new_tokens = []
    for token in tokens:
        if randint(0, 1) == 1:
            # Replace a word with its synonym
            synonyms = wordnet.synsets(token)
            if synonyms:
                token = choice(synonyms).lemmas()[0].name()
        new_tokens.append(token)
    
    # Join the tokens back into a string
    augmented_text = ' '.join(new_tokens)
    
    return augmented_text

# Apply text augmentation to the existing data
augmented_data = []
for index, row in data.iterrows():
    first_context = row['FIRST_CONTEXT']
    last_context = row['LAST_CONTEXT']
    genre = row['GENRE']
    
    # Generate 5 new samples for each existing sample
    for _ in range(5):
        new_first_context = augment_text(first_context)
        new_last_context = augment_text(last_context)
        augmented_data.append([new_first_context, new_last_context, genre])

# Convert the augmented data into a Pandas dataframe
augmented_df = pd.DataFrame(augmented_data, columns=['FIRST_CONTEXT', 'LAST_CONTEXT', 'GENRE', 'AUTHOR', 'TITLE'])

# Append the augmented data to the original data
data = pd.concat([data, augmented_df], ignore_index=True)

# Save the updated data to a new Excel file
data.to_excel("./augmented_dataset.xlsx", index=False)