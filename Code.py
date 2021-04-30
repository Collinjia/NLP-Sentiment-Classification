# -*- coding: utf-8 -*-


"""
This program is a BoW text classifier for sentiment analysis of the reviews.
It analyzes the reviews and classify them to three bins: positive, negative
and neutral.

Created by Ruoxi Jia.
"""

import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer, word_tokenize
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import nltk

# Import the reviews.csv to dataframe
InputData = pd.read_csv('reviews.csv', delimiter='\t')

'''
# Figure out how many data in each Rating.
# Since we have 1465 positive rating, 297 neutral rating and 158 negative rating
# we need to reduce the number of data with positive rating to 300.

positive = InputData[InputData["RatingValue"] > 3]
neutral = InputData[InputData["RatingValue"] == 3]
negative = InputData[InputData["RatingValue"] < 3]

print("we have", positive.shape[0], "positive rating")
print("we have", neutral.shape[0], "neutral rating")
print("we have", negative.shape[0], "negative rating")

'''


# Bin the rating to positive, neutral and negative.
def add_rate(data):
    i = 0
    for i in range(0, data.shape[0]):
        # For the data with RatingValue > 3, assign with "positive" Rating.
        if data.loc[i, "RatingValue"] > 3:
            data.loc[i, "Rating"] = "positive"
        # For the data with RatingValue = 3, assign with "neutral" Rating.
        elif data.loc[i, "RatingValue"] == 3:
            data.loc[i, "Rating"] = "neutral"
        # For the data with RatingValue < 3, assign with "negative" Rating.
        else:
            data.loc[i, "Rating"] = "negative"
    i += 1


# Add rating for InputData
add_rate(InputData)

n_pos = 0

# Reduce the number of positive rating to 300.
for j in range(0, InputData.shape[0]):
    if InputData.loc[j, "Rating"] == "positive":
        n_pos += 1
        if n_pos > 300:
            InputData.loc[j, "Rating"] = "NaN"
    j += 1
InputNew = InputData.drop(InputData[InputData.Rating == "NaN"].index)

# Split the dataset into training and valid
training, valid = train_test_split(InputNew, test_size=0.3, random_state=42)

# Export the training and valid dataset into csv file
training.to_csv("training.csv")
valid.to_csv("valid.csv")

# Load the training data
train = pd.read_csv("training.csv")

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# create a tokenizer with lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        # Remove the stop words and lemmatize the rest.
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if
                t not in nltk.corpus.stopwords.words('english')]


# Build a SGD pipeline
text_clf = Pipeline([
    # Tokenize the text with ngram 1-3 and removing stopwords
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             ngram_range=(1, 3),
                             stop_words='english')),
    # TF-IDF transform
    ('tfidf', TfidfTransformer()),
    # SGD Classification model
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
# Train the model
text_clf.fit(train['Review'], train['Rating'])

# Import the validation data
validation = pd.read_csv("valid.csv")

# Add rating column to validation data
add_rate(validation)

# Implement the model
predicted = text_clf.predict(validation['Review'])

# calculate the accuracy
accuracy = round(np.mean(predicted == validation['Rating']), 2)
# calculate the F1 score
F1_score = round(metrics.f1_score(validation.Rating, predicted, average='weighted'), 2)
# print out the result
print("accuracy:", accuracy, "\n")
print("F1_score:", F1_score, "\n")
print("Confusion_matrix:")
# generate the confusion matrix
conf = pd.DataFrame(metrics.confusion_matrix(validation.Rating, predicted),
                    index=['negative', 'neutral', 'positive'],
                    columns=['negative', 'neutral', 'positive'])
print(conf)
