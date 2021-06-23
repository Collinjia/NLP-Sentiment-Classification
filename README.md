# Sentiment Classification

This project is to build a sentiment classifier to classify restaurant reviews. Data is in [review.csv](review.csv）.

It has two approaches. One is using [**Bag of Words in Sklearn**](BagofWords_Approach.ipynb), the other is using the [**ULMfit in fastai**](Fastai_Approach.ipynb).

## Bag of Words Approach:

  1. Tokenization: change the sentence into words.
  2. Lemmatization: standardize the words, like change went and goes to go. Examples in [here](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/).
  3. Remove stop words: Remove the stop words like the, a, which might influence the model.
  4. TF - IDF transform: Count the term frequency of the word, and calculate term-frequency times inverse document-frequency. Details in [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).
  5. Build the model using SGD classifier

## Fastai Text Approach: 
Fastai.text provides a pretrained NLP model basing on [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset. All you need to do is to fine-tune the pre-trained model on your dataset and make prediction.

ULMFiT achieves good results by relying on techniques like:

 * Discriminative fine-tuning (layer-specific learning rates)
 * Slanted triangular learning rates (increasing and then decreasing learning rates over epochs)
 * Gradual unfreezing (gradually unfreeze layers, starting from the last)

 **Runtime**

 This is deep-learning-NLP, and the harware matters. Colab provides both GPUs (graphics processing units) and TPU (tensor processing units). And if you have a Nvidia GPU, [Nvidia cuda](https://developer.nvidia.com/cuda-toolkit) will help in speed up the process.



