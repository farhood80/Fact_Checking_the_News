#!/usr/bin/env python
# coding: utf-8

# <B> Fake News Detection Project

# In[42]:


#Importing the dependencies

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[43]:


import nltk
nltk.download('stopwords')


# In[44]:


#printing the stopwords in english
print(stopwords.words('english'))


# <B> Data Pre-Processing 

# In[45]:


#loading the dataset and convert it to pandas dataframe
News = pd.read_csv('/home/farhood/Desktop/train.csv')


# In[46]:


#checking the number of the rows and columns in the dataset
News.shape


# In[47]:


#getting inforamtion about the  dataset
News.info()


# In[48]:


#counting the missing values 
News.isnull().sum()


# In[49]:


News.head(10)
# 0 = real news , 1 = fake news


# In[50]:


# replacing the null values with empty string
News = News.fillna('')


# In[51]:


# merging the author  and title columns
News['content'] = News['author']+' '+ News['title']
print(News['content'])


# In[52]:


# separating the data & label
x = News.drop(columns= 'label', axis = 1)
y = News['label']

#if you want see the result 
#print(x)
#print(y)


# <b> stemming
#  
#  <b> If You Want To Know More About The Stemming [<b> What Is Stemming?](https://www.techtarget.com/searchenterpriseai/definition/stemming)   

# In[53]:


port_stem = PorterStemmer()


# In[54]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content  = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[55]:


News['content'] = News['content'].apply(stemming)


# In[58]:


print(News['content'])


# In[59]:


#separating the data and label

x = News['content'].values
y = News['label'].values

#if you want see the result 
#print(x)
#print(y)


# In[61]:


# converting the textual data to numerical data
vect = TfidfVectorizer()
vect.fit(x)
x = vect.transform(x)


# In[62]:


print(x)


# In[66]:


#spliting data set to train and test data

x_train , x_test , y_train , y_test = train_test_split(x , y, test_size = 0.2, stratify=y, random_state =2)


# In[68]:


#training the model with logistic regression

model = LogisticRegression()


# In[69]:


model.fit(x_train, y_train)


# In[70]:


# evalution
#accuracy of the model

# accuracy of the training data

x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)


# In[71]:


print(" the accuracy of the training data is: ", training_data_accuracy)


# In[72]:


# accuracy of the test data

x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)

print(" the accuracy of the test data is: ", test_data_accuracy)


# In[77]:


# builing a predictive system
n = int(input("please enter a neews from the dataset: "))
x_new = x_test[n] # you can input any

prediction = model.predict(x_new)
print(prediction)

if (prediction[0]==0):
  print('This news is Real')
else:
  print('This news is Fake')


# In[ ]:




