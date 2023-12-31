# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, SimpleRNNCell, RNN, Dense,Layer

# %%
max_sentence_length = 100
embedding_dim = 100
max_char_length = 15
char_vocab_size = 36
num_diacritics = 15
lstm_units = 32

# %%
from preprocessing import *
trainSet = readFile('dataset/train.txt')

# %%
sentences_without_diacritics, diacritics = get_sentences(cleanText(trainSet[:1000]))

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
model.save("word2vec_model.bin")

# %%
harakatID   = load_binary('diacritic2id','./')

def get_diacritic_hot_vector(haraka):
    if haraka not in harakatID:
        return list(np.ones(15,dtype=int) )
    vector = [0 for _ in range(len(harakatID))]
    # print("haraka:" + haraka)
    vector[harakatID[haraka] ] = 1
    return vector


# %%
# harakat   = load_binary('diacritic2id','./')
# for i in harakat:
#     print(i,harakat[i])
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
Y.dtype

# %%
model.fit([X_sentence,X_char,X_scalar], Y, epochs= 10)


