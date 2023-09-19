import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, all_words):
    ignore_words = ['?', '!', ',', '.', ';', '/']
    all_words = ([stem(word) for word in all_words if word not in ignore_words])
    tokenized_sentence = [stem(word) for word in tokenized_sentence if word not in ignore_words]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0
    return bag


























# all_words = "My name is Otavie Okuoyo Loveday??"
# tok_sent = "My name is Loveday?"
# a = tokenize(all_words)
# b = tokenize(tok_sent)
# all_words = [stem(w) for w in a]
# tok_sent = [stem(w) for w in b]
# print(all_words)
# print(tok_sent)
# print(bag_of_words(tok_sent, all_words))
# str = "My name is Otavie Okuoyo, and right now I am a Research student at Covenant University, Nigeria. Otavie is a programming and a web developer."
# a = tokenize(str)
# stem_word = [stem(w) for w in a]