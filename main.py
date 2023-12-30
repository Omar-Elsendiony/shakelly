# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, SimpleRNNCell, RNN, Dense,Layer

# %%
# # import tensorflow as tf
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Input, LSTM, Bidirectional, SimpleRNNCell, RNN, Dense,Layer
# class SelectHiddenState(Layer):
#     def __init__(self, **kwargs):
#         super(SelectHiddenState, self).__init__(**kwargs)

#     def call(self, lstm_output, scalar_input):
#         # Ensure scalar_input is an integer for indexing
#         timestep_index = tf.cast(tf.squeeze(scalar_input, axis=-1), tf.int32)
#         # Gather the specific hidden state for each batch
#         selected_state = tf.gather(lstm_output, timestep_index, batch_dims=1, axis=1)
#         return selected_state

# # Example usage with your model
# # lstm_output is from your BiLSTM layer
# # scalar_input is your additional input
# max_sentence_length = 100  # Maximum length of sentence embeddings
# embedding_dim = 100        # Dimension of sentence embeddings
# max_char_length = 15       # Maximum length of a word in characters
# char_vocab_size = 36       # Number of unique characters
# num_diacritics = 15         # Number of possible diacritics for each character, including no diacritic

# # Parameters
# lstm_units = 32

# # Character input
# char_input = Input(shape=(max_char_length, char_vocab_size))

# # Inputs
# sentence_input = Input(shape=(max_sentence_length, embedding_dim))
# scalar_input = Input(shape=(1,), name='scalar_input')

# # BiLSTM layer with return_state
# bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True))
# bi_lstm_output, forward_h, forward_c, backward_h, backward_c = bi_lstm(sentence_input)
# print( forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)
# #, forward_h, forward_c, backward_h, backward_c
# # Average the forward and backward states (or choose another method to combine them)
# select_state_layer = SelectHiddenState()
# hidden_state_nth_timestep = select_state_layer(bi_lstm_output, scalar_input)

# #hidden_state_nth_timestep = bi_lstm_output[:, scalar_input[1], :]
# print(bi_lstm_output.shape)

# # RNN layer with initial state from BiLSTM
# rnn_cell = SimpleRNNCell(64)
# rnn_layer = RNN(rnn_cell, return_sequences=True)
# rnn_output = rnn_layer(char_input,initial_state=hidden_state_nth_timestep)

# # Output layer
# output_layer = Dense(num_diacritics, activation='softmax')(rnn_output)

# # Build and compile the model
# # Assuming sentence_input and scalar_input are defined as Input layers
# model = Model(inputs=[sentence_input, char_input,scalar_input], outputs=output_layer)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Model summary
# model.summary()

# %%
# X = 
# Y = 
from preprocessing import *

trainSet = readFile('dataset/train.txt')

# %%
sentences_without_diacritics, diacritics = get_sentences(trainSet)

# %%
# sentences_without_diacriticsTest, diacriticsTest = get_sentences("يُسَنُّ أَنْ يُصَانَ عَنْ رَائِحَةٍ كَرِيهَةٍ مِنْ بَصَلٍ وَثُومٍ وَكُرَّاتٍ )")

# %%
print(sentences_without_diacritics)

# %%
print(diacritics)

# %%
word2vecmodel = makeWord2VecModel(sentences_without_diacritics)
keys = word2vecmodel.wv.key_to_index
print(keys)

# %%
def getEmbeddingsSentences(sentences, word2vecmodel):
    embeddingSentences = [] # list of all sentences
    keys = word2vecmodel.wv.key_to_index
    for s in sentences:
        embeddingTemp = []  # list for one sentence
        for w in s:
            if w in keys:
                embeddingTemp.append(word2vecmodel.wv[w])
            ### unknown OOV till now
        embeddingSentences.append(embeddingTemp)
    return embeddingSentences
embeddingsSentences = getEmbeddingsSentences(sentences_without_diacritics, word2vecmodel)
print(embeddingsSentences)

# %%
harakat   = load_binary('diacritic2id','./')
for i in harakat:
    print(i,harakat[i])
#يُسَنُّ أَنْ يُصَانَ عَنْ رَائِحَةٍ كَرِيهَةٍ مِنْ بَصَلٍ وَثُومٍ وَكُرَّاتٍ( 3 / 297 )

corpusDiacList = []
for sentence in diacritics:
    sentenceDiacList = []
    for word in sentence:
        #merge each list 
        oneDiacStr = ''
        diacWordList = []
        for diac in word:
            if diac == '_':
                oneDiacStr = ''
            else:    
                oneDiacStr = ''.join(diac)
            #print(oneDiacStr,harakat[oneDiacStr]) 
            diacWordList.append(get_diacritic_hot_vector(oneDiacStr)) 
        sentenceDiacList.append(diacWordList)       
    corpusDiacList.append(sentenceDiacList)
print(corpusDiacList)
        

# %%
def get_diacritic_hot_vector(haraka):
    vector = [0 for _ in range(len(harakat))]
    # print("haraka:" + haraka)
    vector[harakat[haraka] ] = 1
    return vector

print(get_diacritic_hot_vector('َ'))

# %%
def getDiacriticEncoding(wordDi):
    word_embedding = list()
    for w in wordDi:
        if (len(w) > 1):
            # print(ord(w[0]))
            print(ord(w[1]))
            w = w[0]
        word_embedding.append(get_diacritic_hot_vector(w))
    return word_embedding
# getDiacriticEncoding('ًُ')


# %%
# X  = numpy.array([]) # input
# Y  = numpy.array(corpusDiacList, dtype=object) # output

# import numpy as np
# for i in range(len(embeddingsSentences)):
#     for j in range(len(embeddingsSentences[i])):
#         x = numpy.array([])
#         x=np.append(x,numpy.array([embeddingsSentences[i]]))
#         x=np.append(x,[getCharacterEncoding(sentences_without_diacritics[i][j])])
#         x=np.append(x,j)
#         X=np.append(X,x)
#         ### y ###
#         # y = getDiacriticEncoding(diacritics[i][j])
#         # Y=np.append(Y,y)
# print(X.shape)

# %%
# import numpy as np
# import tensorflow as tf

# X_embeddings = []  # List to hold embedding arrays
# X_scalars = []     # List to hold scalar values
# Y  = corpusDiacList # output

# for i in range(len(embeddingsSentences)):
#     for j in range(len(embeddingsSentences[i])):
#         # Add embeddings
#         X_embeddings.append(np.array(embeddingsSentences[i]))

#         # Add scalar values as a tuple or list
#         char_encoding = getCharacterEncoding(sentences_without_diacritics[i][j])
#         X_scalars.append([char_encoding, j])

#         # For Y
#         # y = getDiacriticEncoding(diacritics[i][j])
#         # Y.append(y)

# # Convert to tensors
# X_embeddings_tensor = tf.ragged.constant(X_embeddings)  # Ragged tensor for embeddings
# X_scalars_tensor =  tf.ragged.constant(X_scalars,shape=())      # Regular tensor for scalars
# Y_tensor = tf.convert_to_tensor(Y)

# # Check the shapes
# print(X_embeddings_tensor.shape)
# print(X_scalars_tensor.shape)
# print(Y_tensor.shape)


# %%
# X  = list() # input
# Y  = list() # output
# for i in range(len(embeddingsSentences)):
#     for j in range(len(embeddingsSentences[i])):
#         x = list()
#         x.append(embeddingsSentences[i])
#         x.append(getCharacterEncoding(sentences_without_diacritics[i][j]))
#         x.append(j)
#         X.append(x)
#         ### y ###
#         y = getDiacriticEncoding(diacritics[i][j])
#         Y.append(y)
    
# print(X[0][1])

# %%
# import numpy as np
# corpus=np.array(corpusDiacList, dtype=object).flatten()
# print(corpus.shape)
# corpus.flatten()
# print(corpus.flatten()[2])
#corpusDiacList=np.array(corpusDiacList)
print(corpusDiacList)
flattened = [item for sublist in corpusDiacList for item in sublist]
Y__padded = pad_sequences(flattened, maxlen=max_char_length, padding='post', dtype='float32')
print(Y__padded.shape)

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_sentence = []  # List for sentence_input
X_char = []      # List for char_input
X_scalar = []    # List for scalar_input
flattened = [item for sublist in corpusDiacList for item in sublist]
Y__padded = pad_sequences(flattened, maxlen=max_char_length, padding='post', dtype='float32')
          # List for output

for i in range(len(embeddingsSentences)):
    for j in range(len(embeddingsSentences[i])):
        # Add embeddings
        X_sentence.append(embeddingsSentences[i])

        # Prepare char_input
        char_encoding = getCharacterEncoding(sentences_without_diacritics[i][j])
        # Ensure char_encoding is shaped as (max_char_length, char_vocab_size)
        # This might require reshaping or padding depending on your getCharacterEncoding function
        X_char.append(char_encoding)

        # Scalar input
        X_scalar.append([j])

        # Output
        # y = getDiacriticEncoding(diacritics[i][j])
        # Y.append(y)

# Padding X_sentence
# Assuming max_sentence_length and embedding_dim are defined
X_sentence_padded = pad_sequences(X_sentence, maxlen=max_sentence_length, padding='post', dtype='float32')
X_character_padded = pad_sequences(X_char, maxlen=max_char_length, padding='post', dtype='float32')
# Y__padded = pad_sequences(Y, maxlen=max_char_length, padding='post', dtype='float32')

# Convert lists to numpy arrays
X_sentence = np.array(X_sentence_padded)
X_char = np.array(X_character_padded)
X_scalar = np.array(X_scalar)

Y = np.array(Y__padded)

print(X_sentence.shape, X_char.shape, X_scalar.shape, Y.shape)


# %%
class SelectHiddenState(Layer):
    def _init_(self, **kwargs):
        super(SelectHiddenState, self)._init_(**kwargs)

    def call(self, lstm_output, scalar_input):
        timestep_index = tf.cast(tf.squeeze(scalar_input, axis=-1), tf.int32)
        selected_state = tf.gather(lstm_output, timestep_index, batch_dims=1, axis=1)
        return selected_state

# Parameters
max_sentence_length = 100
embedding_dim = 100
max_char_length = 15
char_vocab_size = 36
num_diacritics = 15
lstm_units = 32

# Inputs
char_input = Input(shape=(max_char_length, char_vocab_size))
sentence_input = Input(shape=(max_sentence_length, embedding_dim))
scalar_input = Input(shape=(1,), name='scalar_input')

# Padding layer for sentence_input (adjust padding as needed)
# sentence_padding_layer = ZeroPadding1D(padding=(1, 1))  # Example padding
# padded_sentence_input = sentence_padding_layer(sentence_input)
# padding_layer = ZeroPadding1D(padding=(1, 1))  # Example padding
# padded_char_input = padding_layer(char_input)
# BiLSTM layer
bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True))
bi_lstm_output, forward_h, forward_c, backward_h, backward_c = bi_lstm(sentence_input)
print(forward_h)
# Select state layer
select_state_layer = SelectHiddenState()
hidden_state_nth_timestep = select_state_layer(bi_lstm_output, scalar_input)

# RNN layer
rnn_cell = SimpleRNNCell(32)
rnn_layer = RNN(rnn_cell, return_sequences=True)
rnn_output = rnn_layer(char_input, initial_state=(forward_c+backward_c)/2)

# Output layer
output_layer = Dense(num_diacritics, activation='softmax')(rnn_output)

# Model
model = Model(inputs=[sentence_input, char_input, scalar_input], outputs=output_layer)
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Summary
model.summary()

# %%

model.fit([X_sentence,X_char,X_scalar], Y, epochs= 10)

# %%



