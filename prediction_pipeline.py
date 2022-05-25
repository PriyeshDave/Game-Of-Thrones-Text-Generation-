import pandas as pd
import numpy as np
import nltk
import re
import tensorflow
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences


#sw = pd.read_csv('./Datasets/StopWords.csv')['StopWords'].to_list()
#nltk.download('wordnet')
#lm = WordNetLemmatizer()
# lm = joblib.load('./Models/Lemmatizer.pkl')
model = tensorflow.keras.models.load_model('./Model/Word2Vec-Model-450-512.h5')
tokenizer = joblib.load('./Model/Tokenizer.pkl')

#one_hot_df = pd.read_csv('./Datasets/One Hot Encoded Data.csv').set_index('Word')

#mapper = {0 : 'Not Toxic', 1 : 'Toxic'}

#voc_size = 16000
#sent_length = 25
#embedding_vector_features = 300

def read_file():
  file = open('./Datasets/Cleaned-Text.txt', 'r')
  text = file.read().strip()
  file.close()
  return text

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # predict probabilities for each word
    yhat = model.predict(encoded, verbose=0)
    # map predicted word index to word
    #print(np.argmax(yhat))
    yhat = np.argmax(yhat)
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        #print(word)
        out_word = word
        break
    # append to input
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

def get_predictions(sentence, n_words):
  generated_sequence = generate_seq(model, tokenizer, 50, sentence, n_words)
  return generated_sequence
  




