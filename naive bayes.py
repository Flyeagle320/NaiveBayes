# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:06:09 2022

@author: Rakesh
"""

##########################problem 1###############################
import pandas as pd
import numpy as np

##loading dataset#
salary_train = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Naive Bayes/SalaryData_Train.csv', encoding='ISO-8859-1')
salary_test = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Naive Bayes/SalaryData_Test.csv', encoding='ISO-8859-1')

##Preparing naives bayes model on train data#
from sklearn.naive_bayes import MultinomialNB as MB

x= {' <=50K':0 , ' >50K' : 1}
salary_train.Salary = [x[item]for item in salary_train.Salary]
salary_test.Salary = [x[item]for item in salary_test.Salary]

##getting dummy value for train data#
salary_train_dummies =pd.get_dummies(salary_train) 
salary_train_dummies.drop(['Salary'], axis= 1 , inplace = True)
salary_train_dummies.head(3)

#check na values#
salary_train_dummies.columns[salary_train_dummies.isna().any()]

##getting dummies for test data#
salary_test_dummies = pd.get_dummies(salary_test)
salary_test_dummies.drop(['Salary'], axis = 1 , inplace = True)
salary_test_dummies.head(3)
#checking for na value #
salary_test_dummies.columns[salary_test_dummies.isna().any()]

##Multinomial naive bayes#
classifier_mb = MB()
classifier_mb.fit(salary_train_dummies, salary_train.Salary)

##Test data evaluation##
test_pred= classifier_mb.predict(salary_test_dummies)
accuracy_test = np.mean(test_pred==salary_test.Salary)
accuracy_test

from sklearn.metrics import accuracy_score
accuracy_score(test_pred, salary_test.Salary)

##Train data evaluation#
train_pred = classifier_mb.predict(salary_train_dummies)
accuracy_test = np.mean(train_pred==salary_train.Salary)
accuracy_test
##Multinomial Naive Bayes changing default alpha for laplace smoothing#
classifier_mb_lap =MB(alpha=3)
classifier_mb_lap.fit(salary_train_dummies,salary_train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(salary_test_dummies)
accuracy_test_lap = np.mean(test_pred_lap==salary_test.Salary)
accuracy_test_lap

accuracy_score(test_pred_lap,salary_test.Salary)

pd.crosstab(test_pred_lap, salary_test.Salary)

##Training accuracy data#
train_pred_lap = classifier_mb_lap.predict(salary_train_dummies)
accuracy_train_lap=np.mean(train_pred_lap==salary_train.Salary)
accuracy_train_lap

################################problem2##################################
import pandas as pd
import numpy as np
##loading dataset#
car = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Naive Bayes/NB_Car_Ad.csv' , encoding = 'ISO-8859-1')

##droping userid as it is nominal data#
car = car.iloc[:,1:]
car.head()
#scaling data##
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
car[['Age','EstimatedSalary']] =scaler.fit_transform(car[['Age','EstimatedSalary']])

##Train and test data split#
from sklearn.model_selection import train_test_split
car_train , car_test = train_test_split(car , test_size= 0.2)

## getting dummy variable for train data#
car_train_dummies = pd.get_dummies(car_train)
car_train_dummies.drop(['Purchased'],axis=1 , inplace = True)
car_train_dummies.head(3)

## getting dummy variable for test data#
car_test_dummies = pd.get_dummies(car_test)
car_test_dummies.drop(['Purchased'], axis = 1 , inplace= True)
car_test_dummies.head(3)

##Preparing naive bayes model on train data#
from sklearn.naive_bayes import MultinomialNB as MB

##Multinomial naive bayes#
classifier_mb  =MB(alpha=3)
classifier_mb.fit(car_train_dummies, car_train.Purchased)

##evaluation on Test Data#
test_pred = classifier_mb.predict(car_test_dummies)
accuracy_test = np.mean(test_pred==car_test.Purchased)
accuracy_test

from sklearn.metrics import accuracy_score
accuracy_score(test_pred, car_test.Purchased)

pd.crosstab(test_pred, car_test.Purchased)

##let check training data accuracy##
train_pred= classifier_mb.predict(car_train_dummies)
accuracy_train=np.mean(train_pred==car_train.Purchased)
accuracy_train

##we can see accuracy is less in both test and train data. in this case let us use Gaussian model as present one is not efficient model#
##Gaussian naive bayes#
from sklearn.naive_bayes import GaussianNB as GB

classifier_mb_lap = GB()
classifier_mb_lap.fit(car_train_dummies, car_train.Purchased)

##Evaluation of Test data on applyin laplace#
test_pred_lap = classifier_mb_lap.predict(car_test_dummies)
accuracy_test_lap= np.mean(test_pred_lap==car_test.Purchased)
accuracy_test_lap

accuracy_score(test_pred_lap, car_test.Purchased)
pd.crosstab(test_pred_lap, car_test.Purchased)

##training data accuracy#
train_pred_lap = classifier_mb_lap.predict(car_train_dummies)
accuracy_train_lap= np.mean(train_pred_lap==car_train.Purchased)
accuracy_train_lap
##as you can see accuracy is 88% which is efficient model ##
#############################Problem 3############################################
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

##loading dataset#
tweet = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Naive Bayes/Disaster_tweets_NB.csv',encoding='ISO-8859-1')
tweet= tweet.iloc[: , 3:5] ##removing other columns as it contain nan value#
#cleaning data#
import re
stop_words =[]
##loading custom built stopwords#
with open('D:/DATA SCIENCE ASSIGNMENT/Datasets_Naive Bayes/stop.txt', "r") as sw:
    stop_words=sw.read()
stop_words = stop_words.split("\n")    

def cleaning_text(i):
    i =re.sub("[^A-Za-z" "]+"," ",i).lower()
    i=re.sub("[0-9" "]+"," ",i)
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
            return (" ".join(w))
tweet.text= tweet.text.apply(cleaning_text)        

##removing empty row#
tweet = tweet.loc[tweet.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
##splitting data into train and test#
from sklearn.model_selection import train_test_split

tweet_train , tweet_test=train_test_split(tweet, test_size=0.2)

##creating matrix for token counts#for text document #
def split_into_words(i):
    return [word for word in i.split(" ")]
    
# Defining the preparation of tweet texts into word count matrix format - Bag of Words##
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet.text)

#Defining BOW for all tweets#
all_tweet_matrix = tweet_bow.transform(tweet.text)

##for training messages #
train_tweet_matrix = tweet_bow.transform(tweet.text)
#for test messages#
test_tweet_matrix = tweet_bow.transform(tweet.text)

 #Learning Term weighting and normalizing on entire tweet
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train tweet
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap


accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap















