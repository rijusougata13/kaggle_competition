#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:25:26 2020

@author: sougata
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import string
from nltk.corpus import stopwords

def text_process(news) :
    """
    1. remove punctuation
    2. remove stop words
    3. return list of clean text words
    """
    
    nopunc = [char for char in news if char not in string.punctuation]
    
    nopunc = "".join(nopunc)
    
    return [words for words in nopunc.split() if words.lower() not in stopwords.words('english')]


(train['text'].head().apply(text_process))
(test['text'].head().apply(text_process))

pipeline = Pipeline({
    ('bow',CountVectorizer(analyzer = text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True))
})

    
pipeline.fit(train['text'],train['target'])

pred = pipeline.predict(test['text'])

submission = pd.DataFrame(pred)
submission = pd.concat([test['id'],submission],axis=1)
submission.columns = ['id','target']

submission.to_csv('submission1234.csv',index=False)