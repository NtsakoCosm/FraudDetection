import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,average_precision_score

fraud_df = pd.read_csv('creditcard.csv')
feautures = fraud_df.drop('Class', axis=1)

target = fraud_df['Class']

x_train, x_test, y_train, y_test = train_test_split(feautures,target, test_size=0.25,random_state=42)  

kbest = ["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]
ind = 1

x_train = x_train.loc[:,kbest[:ind]]

x_test = x_test.loc[:,kbest[:ind]]

train = pd.concat([x_train,y_train],axis=1)
test =  pd.concat([x_test,y_test],axis=1)

positive_means = train[train['Class'] == 1].mean(axis=0).drop('Class')
negative_means = train[train['Class'] == 0].mean(axis=0).drop('Class')

postive_std = train[train['Class'] == 1].std(axis=0).drop('Class')
negative_std = train[train['Class'] == 0].std(axis=0).drop('Class')



def GaussianNB(sigma,mean,X):
    classprob = []
    for i,item in sigma.iteritems():
        prob = (1/(np.sqrt(2*np.pi*item)) )*np.exp(-((X[i]-mean[i])**2/(2*item)))
        classprob.append(prob)
    return sum(classprob)

tp = 0
fp = 0
fn = 0
tn = 0
predicted = []
for i,instance in x_test.iterrows():
    pos = GaussianNB(postive_std,positive_means,instance)
    neg = GaussianNB(negative_std,negative_means,instance)
    pos_prob = pos/(pos+neg)
    neg_prob = neg/(pos+neg)
    guess = 1 if pos_prob > neg_prob else 0
    predicted.append(guess)
    if y_test[i] == 1 and guess == 1:
        tp+=1
    if y_test[i] == 1 and guess == 0: 
        fn+=1
    if y_test[i] == 0 and guess == 0:
        tn+=1
    if y_test[i] == 0 and guess == 1:
        fp+=1

def Gauss():
    st.header('Gaussian Naive Bayes')
    gauss ='''
    Much like Logistic Regression, Gaussian Naive Bayes has similar principles based on probability.
    But the differences are very apparent.
    This algorithm is the continuous feature version of the categorical Naive Bayes :
    '''
    st.write('')
    st.subheader('Categorical: ')
    st.image(Image.open('Production/Algorithms/NaiveBayes/nb.jpeg'))
    st.subheader('Continuous')
    st.image(Image.open('Production/Algorithms/NaiveBayes/gnb.png'))
    st.write('Since all my features of the dataset are Continuous, i will be spending all the time on Gaussian Naive Bayes.')
    gnb = '''
The first time gaussian naive clicked in my head was when i ignored the usual Naive bayes(categorical) 
and focused on what the Gaussian Naive bayes Formula was trying to tell me.
    The algorithm is very much like a "statistical test"(everything in ML is stats and math).
    If you have the standard deviation and the mean, you can plot a distribution , then after determine the likelihood of data point belonging in said distribution.
    First off we split the data set based on the target variable
    (fraud/non-fraud) , calculate the mean and the standard deviation for both classes for every variable. 
    after that we calculate the probability of a given instance of belonging in a given class.
    The class with the highest probability is then the label of the instance variable, 
    but then this is done for every variable ,
    so what happens is that the probability of a instance belonging in a given class is the probability of each and 
    every feature value for that instance belonging to a given class, added up and then making a prediction based on the highest class probability.

    I had the shock of my life 
    when the most highly correlated feature: V17 did better alone , 
    then with all the features adding to to. It seems like adding more features decreased my recall score.
    '''
    st.write(gnb)
    st.subheader('My code: ')
    st.code('''
    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,average_precision_score

fraud_df = pd.read_csv('creditcard.csv')
feautures = fraud_df.drop('Class', axis=1)
target = fraud_df['Class']

x_train, x_test, y_train, y_test = train_test_split(feautures,target, test_size=0.25,random_state=42)  

kbest = ["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]
ind = 6

x_train = x_train.loc[:,kbest[:ind]]

x_test = x_test.loc[:,kbest[:ind]]

train = pd.concat([x_train,y_train],axis=1)
test =  pd.concat([x_test,y_test],axis=1)

positive_means = train[train['Class'] == 1].mean(axis=0).drop('Class')
negative_means = train[train['Class'] == 0].mean(axis=0).drop('Class')

postive_std = train[train['Class'] == 1].std(axis=0).drop('Class')
negative_std = train[train['Class'] == 0].std(axis=0).drop('Class')



def GaussianNB(sigma,mean,X):
    classprob = []
    for i,item in sigma.iteritems():
        prob = (1/(np.sqrt(2*np.pi*item)) )*np.exp(-((X[i]-mean[i])**2/(2*item)))
        classprob.append(prob)
    return sum(classprob)

tp = 0
fp = 0
fn = 0
tn = 0
predicted = []
for i,instance in x_test.iterrows():
    pos = GaussianNB(postive_std,positive_means,instance)
    neg = GaussianNB(negative_std,negative_means,instance)
    pos_prob = pos/(pos+neg)
    neg_prob = neg/(pos+neg)
    guess = 1 if pos_prob > neg_prob else 0
    predicted.append(guess)
    if y_test[i] == 1 and guess == 1:
        tp+=1
    if y_test[i] == 1 and guess == 0: 
        fn+=1
    if y_test[i] == 0 and guess == 0:
        tn+=1
    if y_test[i] == 0 and guess == 1:
        fp+=1

    ''')
    
    st.image(Image.open('Production/Algorithms/NaiveBayes/rocnb.png'))
    st.image(Image.open('Production/Algorithms/NaiveBayes/rocsore.png'))
    
    st.image(Image.open('Production/Algorithms/NaiveBayes/prcurve.png'))
    st.image(Image.open('Production/Algorithms/NaiveBayes/auprc.png'))