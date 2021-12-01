import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

fake = pd.read_csv('F:\\Lecture notes\\Sem-5\\MP-III\\Dataset\\Fake.csv')
real = pd.read_csv('F:\\Lecture notes\\Sem-5\\MP-III\\Dataset\\True.csv')

unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        record = row.split('-', maxsplit = 1)
        record[1]
        assert(len(record[0])<120)
    except:
        unknown_publishers.append(index)

real = real.drop(8970, axis = 0)

publisher = []
tmp_text = []

for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('Unknown')

    else:
        record = row.split('-', maxsplit = 1)
        publisher.append(record[0].strip())
        tmp_text.append(record[1].strip())

real['publisher'] = publisher
real['text'] = tmp_text

empty_fake_index = [ index for index,text in  enumerate (fake.text.tolist()) is str(text).strip() == ""]

real['text'] = real['title'] + " " + real['text']
fake['text'] = fake['title'] + " " + fake['text']

real['text'] = real['text'].apply(lambda x: str(x).lower())
fake['text'] = fake['text'].apply(lambda x: str(x).lower())

real['class'] = 1
fake['class'] = 0

real = real[['text', 'class']]
fake = fake[['text', 'class']]
data = real.append(fake, ignore_index=True)

!pip install spacy==2.2.3
!python -m spacy download en_core_web_sm
!pip install BeautifulSoup4==4.9.1
!pip install textblob==0.15.3
!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force--reinstall

import preprocess_kgptalkie as ps
data['text'].apply(lambda x: ps.remove_special_chars(x))

import gensim
y = data['class'].values
X = [d.split() for d in data['text'].tolist()]
DIM = 100
w2v_model = gensim.models.Word2Vec(sentences = X, size = DIM, window = 10, min_count = 1)