import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

lemmatizer = WordNetLemmatizer()

# Load JSON
intents = json.loads(open('AIchatbot.json').read())

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []

    for w in words:
        bag.append(1 if w in doc[0] else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training, dtype=object)

X = list(training[:,0])
y = list(training[:,1])

# Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(np.array(X), np.array(y), epochs=200, batch_size=5)

model.save('model.h5')

print("✅ Model trained successfully")