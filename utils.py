import os
import string
import re   
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')


#convert using chr(harakat[0])
harakat   = [1614,1615,1616,1618,1617,1611,1612,1613]
connector = 1617


def save_binary(data, file, folder):
    location  = os.path.join(folder, (file+'.pickle') )
    with open(location, 'wb') as ff:
        pickle.dump(data, ff, protocol=pickle.HIGHEST_PROTOCOL)

def load_binary(file, folder):
    location  = os.path.join(folder, (file+'.pickle') )
    with open(location, 'rb') as ff:
        data = pickle.load(ff)

    return data




# print(get_sentences("لله \nل \n والمنة"))

def clear_punctuations(text):
    text = "".join(c for c in text if c not in string.punctuation)
    return text

def clear_english_and_numbers(text):
     text = re.sub(r"[a-zA-Z0-9٠-٩]", " ", text)
     return text

def is_tashkel(text):
    return any(ord(ch) in harakat for ch in text)

def clear_tashkel(text):
    text = "".join(c for c in text if ord(c) not in harakat)
    return text

def get_harakat():
    return "".join(chr(item)+"|" for item in harakat)[:-1]


# append _ if no tashekeel for the 7arf
# otherwise, append the tashkeel itself 
# when you loop, the haraka comes before the 7arf 

def get_tashkel(sentence):
    output = []
    current_haraka = ""
    chIndex = 0
    mode = 0  # mode 0 is character meant (expecting to get character) and mode 1 is tashkeel ment (expecting to get tashkeel)
    while chIndex < (len(sentence)):
        characterTashkeels = []
        if mode == 1:
            while chIndex < (len(sentence)) and ord(sentence[chIndex]) in harakat:
                characterTashkeels.append(sentence[chIndex])
                chIndex += 1
            # else:
            #     if current_haraka == "":
            #         current_haraka = "_"
            #     #print("7arf",current_haraka)
            #     characterTashkeels.append(current_haraka)
                # output.append(current_haraka)
                # current_haraka = ""
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
# def get_tashkel(sentence):
#     output = []
#     current_haraka = ""
    


#print(get_tashkel("اً,"))

def combine_text_with_harakat(input_sent, output_sent):
    #print("input : " , len(input_sent))
  
    #fix combine differences
    input_length  = len(input_sent)
    output_length = len(output_sent) # harakat_stack.size()
    for index in range(0,(input_length-output_length)):
        output_sent.append("")

    #combine with text
    text = ""
    for character, haraka in zip(input_sent, output_sent):
        if haraka == '<UNK>' or haraka == '_':
            haraka = ''
        text += character + "" + haraka

    return text

arabic_alphabet_set=load_binary('arabic_letters','./')
arabic_alphabet = dict(zip(arabic_alphabet_set, list(range(len(arabic_alphabet_set)))))
print(arabic_alphabet)

print("length=")
print(len(arabic_alphabet))
#diacritic:id
#harakat=load_binary('diacritic2id','./')


def get_char_vector(char):
    if char in arabic_alphabet:
        vector = [0 for _ in range(len(arabic_alphabet))]
        vector[arabic_alphabet[char] ] = 1
        return vector
    else:
        return list(np.ones(len(arabic_alphabet),dtype=int) )
    


#harakat   = {1614:1,1615:2,1616:3,1618:4,1617:5,1611:6,1612:7,1613:8, 95:9}

# def get_diacritic_hot_vector(haraka):
#     if haraka not in harakat:
#         return list(np.ones(15,dtype=int) )
#     vector = [0 for _ in range(len(harakat))]
#     vector[harakat[ord(haraka)] - 1] = 1
#     return vector 

# print(get_diacritic_hot_vector('ّ'))

# print(arabic_alphabet.keys())

# print(get_char_vector('ؤ'))


# print(combine_text_with_harakat("ال",['_',""]))
