# Sentiment Classification

This project is to build a sentiment classifier to classify restaurant reviews.

It has two approaches. One is using **Bag of Words in Sklearn**, the other is using the **AWD_LSTM in fastai**.

For Bag of Words Approach: I created a pipeline including the following functions: 

  1. Tokenization: change the sentence into words.
  2. Lemmatization: standardize the words, like change went and goes to go. Examples in [here](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/).
  3. Remove stop words: Remove the stop words like the, a, which might influence the model.
  4. TF - IDF transform: Count the term frequency of the word, and calculate term-frequency times inverse document-frequency. Details in [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).
  5. Build the model using SGD classifier



