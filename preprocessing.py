# %%
from utils import *

# %%
import nltk
import string
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')

# %%
# read the training set
file_path = "dataset/train.txt"
# Open the file for reading
with open(file_path, 'r', encoding='utf-8') as file:
    # Read the content of the file
    file_content = file.read()

# %%
def get_sentences(data):
    #return nltk.tokenize.sent_tokenize(data) #sent_tokenize(data.strip())
    sentences = re.split("[\n.،]+", data)
    # sentences = [s for sentence in senten]
    words_per_sentence = []
    tashkeel_per_sentence = []
    for sentence in sentences:
        if sentence == '': continue # new line may be followed by full stop
        sentence = clear_punctuations(sentence)
        # tashkel, length = get_tashkel(sentence)
        # sentence = clear_tashkel(sentence)
        sentence = word_tokenize(sentence.strip())
        words_per_sentence.append(sentence)
        # temp_tashkeel = []
        # for word in sentence:
        #     temp_tashkeel.append(get_tashkel(word))
    
    tashkeel_per_sentence = [[None for _ in w ]for w in words_per_sentence]
    for i, x in enumerate(words_per_sentence):
        for j, y in enumerate(x):
            tashkeel = get_tashkel(y)
            words_per_sentence[i][j] = clear_tashkel(y)
            tashkeel_per_sentence[i][j] = tashkeel
    # return [sent for sentence in sif line for sent in sent_tokenize(line.strip()) if sent]
    #return [sent for line in data.split('\n') if line for sent in sent_tokenize(line) if sent]
    return words_per_sentence, tashkeel_per_sentence

# %%
file_content = clear_english_and_numbers(file_content)

# %%
# words_without_diacritics = clear_tashkel(file_content)

# %%
# test sentence
test_sentence = '''
قَوْلُهُ : ( أَوْ قَطَعَ الْأَوَّلُ يَدَهُ إلَخْ ) قَالَ الزَّرْكَشِيُّ( 14 / 123 )
ابْنُ عَرَفَةَ : قَوْلُهُ : بِلَفْظٍ يَقْتَضِيه كَإِنْكَارِ غَيْرِ حَدِيثٍ بِالْإِسْلَامِ وُجُوبَ مَا عُلِمَ وُجُوبُهُ مِنْ الدِّينِ ضَرُورَةً ( كَإِلْقَاءِ مُصْحَفٍ بِقَذَرٍ وَشَدِّ زُنَّارٍ ) ابْنُ عَرَفَةَ : قَوْلُ ابْنِ شَاسٍ : أَوْ بِفِعْلٍ يَتَضَمَّنُهُ هُوَ كَلُبْسِ الزُّنَّارِ وَإِلْقَاءِ الْمُصْحَفِ فِي صَرِيحِ النَّجَاسَةِ وَالسُّجُودِ لِلصَّنَمِ وَنَحْوِ ذَلِكَ ( وَسِحْرٍ ) مُحَمَّدٌ : قَوْلُ مَالِكٍ وَأَصْحَابِهِ أَنَّ السَّاحِرَ كَافِرٌ بِاَللَّهِ تَعَالَى قَالَ مَالِكٌ : هُوَ كَالزِّنْدِيقِ إذَا عَمِلَ السِّحْرَ بِنَفْسِهِ قُتِلَ وَلَمْ يُسْتَتَبْ .
'''

# %%
dataset_without_diacritic, diacritics = get_sentences(clear_english_and_numbers(test_sentence))

# %%
test2 = 'قوله   أو قطع الأول يده إلخ  قال الزركشي'
print(word_tokenize(test2.strip())[0])

# %%
from gensim.models import Word2Vec
# Define and train Word2Vec model

word2vec_model = Word2Vec(sentences=dataset_without_diacritic, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model (optional)
word2vec_model.save("word2vec_model.model")

model_keys = word2vec_model.wv.key_to_index
# print(word2vec_model.wv.vectors.shape)
if 'وله' in model_keys: print(word2vec_model.wv['ق'])

# %%
test3 = 'قوله   او قطع الأول يده إلخ  قال الزركشي'

get_char_vector(word_tokenize(test2.strip())[1][1])

# %%
import numpy

def getCharacterEncoding(word):
    word_embedding = list()
    for w in word:
        word_embedding.append(get_char_vector(w))
    return word_embedding

getCharacterEncoding('ألزركش')


