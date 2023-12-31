# %% [markdown]
# ## This module purpose is to extract the features of the words and the characters (if we want)

# %% [markdown]
# #### TfIdf feature extraction

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
def tfidfFeatureExtraction():
    file_path = "dataset/train.txt"
    # Open the file for reading
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the content of the file
        file_content = file.read()

    # Separate based on the delimiter '.'
    sentences = file_content.split('.')

    # Remove empty strings from the list (resulting from consecutive '.' characters)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]


    # what is tfidf vectorizerr
    # it is a combination between tfidf CountVectorizer that calculated the count
    # and tfidf transform that normalizes the results
    vectorizer = TfidfVectorizer(encoding='utf-8')

    # what is fit and transform?
    # fit extracts the count of the instances of this word in this document
    # then the vector (column or row) includes then th
    # transform nomalizes the data so that the data values are between 0 and 1
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_matrix_dense = tfidf_matrix.todense()
    # Get the feature names (terms) so that we can search for this word vector (word or term consecu)
    feature_names = vectorizer.get_feature_names_out()

    # search word
    search_word = "10"
    if search_word in feature_names:
        word_index = list(feature_names).index(search_word)
        print(tfidf_matrix[:, word_index])  # the first dim is the documents , the second is the word or term


# %% [markdown]
# #### Word2Vec feature extraction

# %%
from gensim.models import Word2Vec
def word2VecFeatureExtraction():
    # Define and train Word2Vec model

    word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Save the trained model (optional)
    word2vec_model.save("word2vec_model.model")


# %% [markdown]
# #### Bag of words

# %%
def bagOfWordsFeatureExtraction():
    vectorizer = CountVectorizer()

    # Fit and transform the training set
    X = vectorizer.fit_transform(sentences)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Convert the matrix to a dense array and display the result
    dense_array = X.toarray()


