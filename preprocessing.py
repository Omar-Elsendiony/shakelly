from utils import *
import nltk
import string
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize

# read the training set
def readFile(file_path):
    # Open the file for reading
    file_content = None
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the content of the file
        file_content = file.read()
    return (file_content)

def cleanText(data):
    punctuations=["،","؛","؟",".",",","!",":"," "]
    for char in data:
        #check if char is not in arabic alphabet or harakat
        if char not in (tuple(list(harakat)+list(arabic_alphabet.keys())+punctuations)):
            data = data.replace(char,'')
    # split text into sentences using regex
    
    data = re.sub('  +',' ', data)
    
    data = re.split(r'[،؛؟.,!:]+', data)
    
    return data
def get_sentences(sentences):
    words_per_sentence = []
    tashkeel_per_sentence = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence == '': continue # new line may be followed by full stop
        sentence = word_tokenize(sentence)
        words_per_sentence.append(sentence)
    
    tashkeel_per_sentence = [[None for _ in w ]for w in words_per_sentence]
    for i, x in enumerate(words_per_sentence):
        for j, y in enumerate(x):
            tashkeel = get_tashkel(y)
            words_per_sentence[i][j] = clear_tashkel(y)
            tashkeel_per_sentence[i][j] = tashkeel
    return words_per_sentence, tashkeel_per_sentence

sent=["عَُمرً"] 

print(get_sentences(sent))

test_sentence = '''
قَوْلُهُ : ( أَوْ قَطَعَ الْأَوَّلُ يَدَهُ إلَخْ ) قَالَ الزَّرْكَشِيُّ( 14 / 123 )
ابْنُ عَرَفَةَ : قَوْلُهُ : بِلَفْظٍ يَقْتَضِيه كَإِنْكَارِ غَيْرِ حَدِيثٍ بِالْإِسْلَامِ وُجُوبَ مَا عُلِمَ وُجُوبُهُ مِنْ الدِّينِ ضَرُورَةً ( كَإِلْقَاءِ مُصْحَفٍ بِقَذَرٍ وَشَدِّ زُنَّارٍ ) ابْنُ عَرَفَةَ : قَوْلُ ابْنِ شَاسٍ : أَوْ بِفِعْلٍ يَتَضَمَّنُهُ هُوَ كَلُبْسِ الزُّنَّارِ وَإِلْقَاءِ الْمُصْحَفِ فِي صَرِيحِ النَّجَاسَةِ وَالسُّجُودِ لِلصَّنَمِ وَنَحْوِ ذَلِكَ ( وَسِحْرٍ ) مُحَمَّدٌ : قَوْلُ مَالِكٍ وَأَصْحَابِهِ أَنَّ السَّاحِرَ كَافِرٌ بِاَللَّهِ تَعَالَى قَالَ مَالِكٌ : هُوَ كَالزِّنْدِيقِ إذَا عَمِلَ السِّحْرَ بِنَفْسِهِ قُتِلَ وَلَمْ يُسْتَتَبْ .
'''




# %%
from gensim.models import Word2Vec
# Define and train Word2Vec model
def makeWord2VecModel(dataset):
    word2vec_model = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=1, workers=4)
    # Save the trained model (optional)
    word2vec_model.save("word2vec_model.model")
    return word2vec_model



def getCharacterEncoding(word):
    word_embedding = list()
    for w in word:
        word_embedding.append(get_char_vector(w))
    return word_embedding



