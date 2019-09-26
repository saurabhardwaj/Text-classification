#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


# preparing the machine by loading some libraries
get_ipython().system('pip install tensorflow')


# In[3]:


import tensorflow as tf


# In[5]:


import pandas as pd
import numpy as np


# In[7]:


os.getcwd()


# In[8]:


os.chdir('C:\\Users\\bhardwaj\\Desktop')


# In[9]:


trainDF=pd.read_csv('bbc-text.csv')


# In[10]:


trainDF.head()


# In[13]:


get_ipython().system('pip install -U scikit-learn')


# In[14]:


get_ipython().system('pip install keras')


# In[15]:


get_ipython().system('pip install -U textblob')


# In[16]:


from tensorflow import keras


# In[17]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


# In[18]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['category'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[19]:


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[22]:


def train_model(classifier,xtrain_count,train_y,xvalid_count, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(xtrain_count,train_y)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(xvalid_count)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[23]:


# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: ", accuracy)


# In[ ]:




