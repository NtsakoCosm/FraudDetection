import numpy as np
import streamlit as st
from PIL import Image


predicted_classes = []
def logistic_regression(X,y,xt,yt,epochs=5,learning_rate=0.01,bias=0):
    rows,cols = X.shape
    feature_weights = np.zeros(cols)

    for _ in range(1,epochs):
        for i,instance in X.iterrows():
            predicted = 1/(1+np.exp(-(np.dot(feature_weights,instance)+bias)))
            rule = learning_rate*(y[i]-predicted)*predicted*(1-predicted)
            bias += rule
            feature_weights += rule*instance
    def log_predict(Xtest,ytest):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i,testitem in Xtest.iterrows():
            _predicted = np.dot(testitem,feature_weights)
            binary_output = np.where(_predicted>=0.5,1,0)
            predicted_classes.append(binary_output)
            if ytest[i] == 1 and binary_output == 1:
                tp +=1
            elif ytest[i] == 1 and binary_output == 0:
                fn += 1
            elif ytest[i] == 0 and binary_output == 0 :
                tn +=1
            elif ytest[i] == 0 and binary_output == 1 :
                fp+=1
        #print(f'{tp},{tn},{fp},{fn}, recall = {tp/(tp+fn)}')
        
    log_predict(xt,yt)
    
#logistic_regression(x_train,y_train,x_test,y_test)
#predicted = np.array(predicted_classes)
def LogisticRegProduction():
    st.header('Logistic Regression: ')
    #st.subheader('')
    log = '''
Logistic Regression is a binary classification algorithm. For logistic regression to work we use Maximum likelihood,
    in order to predict,
    first we must estimate the beta variables, then when we have the estimates-The probability a given instance belongs to a given class(fraud/notfraud) ,
    the highest class probability estimation is then chosen over the lower one. 
    All the instances with the highest probability are collected then are entered into the binary cross entropy loss function.
    This then helps us measure how good our model is. 
    If entropy is seen as uncertainty then we can look at the binary cross entropy loss as a way to measure how certain our model is when it is performing predictions. '''
    st.write(log)
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/log.png'))
    sigmoid = '''
The Logistic Regression formula that we use to create a update rule is the one above. 
    We use this formula with gradient descent. Updating the betas is done by setting your betas to zero, plugging in an instance of the data, calculate the prediction, then the error that comes from the cost function is then multiplied by the learning rate , the weights are updated by multiplying the product of the learning rate and the cost function , to the instance itself.
    We do this because we are trying to make our model to fit the gaps of the errors with every instance , but a specific rate (Learning rate).
    The answer(s) then become our betas(weights). Our formula:
    b = b + alpha * (y - prediction) * prediction * (1 - prediction) x
    '''
    st.write(sigmoid)
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/log2.jpg'))
    st.subheader('Logistic Regression from scratch: ')
    st.code('''
    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve
import streamlit as st
from PIL import Image

fraud_df = pd.read_csv('creditcard.csv')
feautures = fraud_df.drop('Class', axis=1)
target = fraud_df['Class']

x_train, x_test, y_train, y_test = train_test_split(feautures,target, test_size=0.25,random_state=42)  

x_train = x_train.loc[:,["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]]

x_test = x_test.loc[:,["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]]

predicted_classes = []
def logistic_regression(X,y,xt,yt,epochs=5,learning_rate=0.01,bias=0):
    rows,cols = X.shape
    feature_weights = np.zeros(cols)

    for _ in range(1,epochs):
        for i,instance in X.iterrows():
            predicted = 1/(1+np.exp(-(np.dot(feature_weights,instance)+bias)))
            rule = learning_rate*(y[i]-predicted)*predicted*(1-predicted)
            bias += rule
            feature_weights += rule*instance
    def log_predict(Xtest,ytest):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i,testitem in Xtest.iterrows():
            _predicted = np.dot(testitem,feature_weights)
            binary_output = np.where(_predicted>=0.5,1,0)
            predicted_classes.append(binary_output)
            if ytest[i] == 1 and binary_output == 1:
                tp +=1
            elif ytest[i] == 1 and binary_output == 0:
                fn += 1
            elif ytest[i] == 0 and binary_output == 0 :
                tn +=1
            elif ytest[i] == 0 and binary_output == 1 :
                fp+=1
        #print(f'{tp},{tn},{fp},{fn}, recall = {tp/(tp+fn)}')
        
    log_predict(xt,yt)
    
logistic_regression(x_train,y_train,x_test,y_test)
predicted = np.array(predicted_classes)
    ''')
    st.subheader('1. Logistic Regression Metrics:')
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/logisticroc.png'))
    
    
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/roclog.png'))
    
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/logisticprerec.png'))
    st.image(Image.open('Fraud/Production/Algorithms/LogisticRegression/auprc.png'))

    st.write('As we can see, logistic regression had a horrible AUPRC score,this tells me that logistic regression can only perform well only when it loses percision(since we are concerned with the recall score.')
    st.write('This also shows how important it is to have different metrics based on different contexts , because if you think about it Logistic Regression had the best AUC score , but the worst AUPRC score.')
    

