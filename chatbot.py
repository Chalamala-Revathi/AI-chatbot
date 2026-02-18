import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
import random

model = load_model('model.h5')
intents = json.loads(open('AIchatbot.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    bag = [0]*len(words)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(msg):
    bow = bag_of_words(msg)
    res = model.predict(np.array([bow]))[0]
    tag = classes[np.argmax(res)]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Test in terminal
while True:
    message = input("You: ")
    print("Bot:", get_response(message))
