import os
import string
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')


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
    for ch in reversed(sentence):
        if ord(ch) in harakat:
            if (current_haraka == "") or\
            (ord(ch) == connector and chr(connector) not in current_haraka) or\
            (chr(connector) == current_haraka):
                current_haraka += ch
            #print(" haraka ", current_haraka)
        else:
            if current_haraka == "":
                current_haraka = "_"
            #print("7arf",current_haraka)
            output.insert(0, current_haraka)
            current_haraka = ""

    return output


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


#print(combine_text_with_harakat("ال",['_',""]))
