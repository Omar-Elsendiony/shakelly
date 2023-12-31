import os
import string
import re   
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')




def save_binary(data, file, folder):
    location  = os.path.join(folder, (file+'.pickle') )
    with open(location, 'wb') as ff:
        pickle.dump(data, ff, protocol=pickle.HIGHEST_PROTOCOL)

def load_binary(file, folder):
    location  = os.path.join(folder, (file+'.pickle') )
    with open(location, 'rb') as ff:
        data = pickle.load(ff)
    return data

harakat= load_binary('diacritics','./')





def clear_tashkel(text):
    text = "".join(c for c in text if c not in harakat)
    return text





def get_tashkel(sentence):
    output = []
    current_haraka = ""
    chIndex = 0
    mode = 0  # mode 0 is character meant (expecting to get character) and mode 1 is tashkeel ment (expecting to get tashkeel)
    while chIndex < (len(sentence)):
        characterTashkeels = []
        if mode == 1:
            while chIndex < (len(sentence)) and sentence[chIndex] in harakat:
                characterTashkeels.append(sentence[chIndex])
                chIndex += 1

            if (len(characterTashkeels) != 0):
                output.append(characterTashkeels)
            else:
                output.append("_") # no tashkeel for now
            # chIndex += 1
            mode = 0
        else:
            mode = 1
            chIndex += 1
    
    if mode == 1: # now I am exepcting tashkeel but the word ended before I find one
        output.append("_")  # _ symbolizes no tashkeel
    return output



arabic_alphabet_set=load_binary('arabic_letters','./')
arabic_alphabet = dict(zip(arabic_alphabet_set, list(range(len(arabic_alphabet_set)))))






def get_char_vector(char):
    if char in arabic_alphabet:
        vector = [0 for _ in range(len(arabic_alphabet))]
        vector[arabic_alphabet[char] ] = 1
        return vector
    else:
        return list(np.ones(len(arabic_alphabet),dtype=int) )
    


