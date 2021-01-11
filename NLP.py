import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

def filter_str(words):
    return re.sub('[^a-zA-Z]',' ',words).split()


def filter_data(review):
    review=[ps.stem(word) for word in filter_str(review.lower()) 
            if word not in set(stopwords.words('English')) or word=='not']
    return ' '.join(review)

data=dataset.iloc[:,0]
data=data.apply(filter_data)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(data).toarray()
y=dataset.iloc[:,-1].values
