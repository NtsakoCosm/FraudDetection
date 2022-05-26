import pandas as pd

import numpy as np

import streamlit as st
from PIL import Image



def p_algo(X,y,xtest,ytest,epochs=3,learning_rate=0.01,bias=0):
    row,cols = X.shape
    feature_weights = np.zeros(cols)
    for _ in range(1,epochs+1):
        for index,feature_vector in X.iterrows():
           linear_output = np.dot(feature_weights,feature_vector.values) + bias
           binary_output = np.where(linear_output>=0, 1,0)
           update_rule = learning_rate*(y[index]-binary_output)
           bias += update_rule
           feature_weights += update_rule*feature_vector
    
    
    def p_algo_fit(Xtest,ytest):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i,instance in Xtest.iterrows():
            y_output = np.dot(instance,feature_weights) + bias
            binary_output_y = np.where(y_output>=0,1,0)
            #predicted.append(binary_output_y)
            if ytest[i] == 1 and binary_output_y == 1:
                tp +=1
            elif ytest[i] == 1 and binary_output_y == 0:
                fn += 1
            elif ytest[i] == 0 and binary_output_y == 0 :
                tn +=1
            elif ytest[i] == 0 and binary_output_y == 1 :
                fp+=1
             
        #print(f'{tp},{tn},{fp},{fn}, recall = {tp/(tp+fn)}')


def perceptron():
    st.header('The Perceptron')
    st.write('The perceptron is a mathematical model of a biological neuron')
    st.write(''' 
    The model is achieved by multiplying a weight vector with a feature vector (with the bias as part of the features), 
    when we are considerng just one instance , the weights(gotten from training) are multiplied by the input values of said instance,
    if the output >= 0 the classification is positive, otherwise negative.
    This is called a step function(Activation function).
    ''')
    imgmodel = Image.open('Production\Algorithms\Perceptron\percptron.png')
    st.image(imgmodel)
    st.write('The weights determine how much each feature impacts the class.')
    st.subheader('The Perceptron from sctratch: ')
    percep = '''
    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve, average_precision_score


fraud_df = pd.read_csv('creditcard.csv')
feautures = fraud_df.drop('Class', axis=1)
target = fraud_df['Class']

x_train, x_test, y_train, y_test = train_test_split(feautures,target, test_size=0.25,random_state=42)  


x_train = x_train.loc[:,["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]]

x_test = x_test.loc[:,["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V9"]]

predicted = []
def p_algo(X,y,xtest,ytest,epochs=3,learning_rate=0.01,bias=0):
    row,cols = X.shape
    feature_weights = np.zeros(cols)
    for _ in range(1,epochs+1):
        for index,feature_vector in X.iterrows():
           linear_output = np.dot(feature_weights,feature_vector.values) + bias
           binary_output = np.where(linear_output>=0, 1,0)
           update_rule = learning_rate*(y[index]-binary_output)
           bias += update_rule
           feature_weights += update_rule*feature_vector
    
    
    def p_algo_fit(Xtest,ytest):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i,instance in Xtest.iterrows():
            y_output = np.dot(instance,feature_weights) + bias
            binary_output_y = np.where(y_output>=0,1,0)
            predicted.append(binary_output_y)
            if ytest[i] == 1 and binary_output_y == 1:
                tp +=1
            elif ytest[i] == 1 and binary_output_y == 0:
                fn += 1
            elif ytest[i] == 0 and binary_output_y == 0 :
                tn +=1
            elif ytest[i] == 0 and binary_output_y == 1 :
                fp+=1
             
        #print(f'{tp},{tn},{fp},{fn}, recall = {tp/(tp+fn)}')
    p_algo_fit(xtest,ytest)

p_algo(x_train,y_train,x_test,y_test)
predicted = np.array(predicted)

    '''
    st.code(percep)
    st.image(Image.open('Production/Algorithms/Perceptron/pre_rec.png'))

    st.image(Image.open('Production/Algorithms/Perceptron/perceptron_auc.png'))
    
    
    st.image(Image.open('Production/Algorithms/Perceptron/percpetronroc.png'))
    st.image(Image.open('Production/Algorithms/Perceptron/auprc.png'))
    
