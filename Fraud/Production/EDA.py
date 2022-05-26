from matplotlib import projections
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image
import plotly.express as px


def Eda():
    pair = Image.open('Production\pairplot.png')
    st.header('Exporatory Data Analysis')
    st.image(Image.open('head.png'))
    st.subheader('Multivarient Analysis')
    eda_desc = '''
    This is the part of the cycle where patterns are displayed as to 
    hypothesize which machine learning algorithm would best suite
    the data.
    One could just train all the algorithms and pick the best one, but as data increases and computation becomes expensive, one must develop a 
    sixth sense(experience) towards what type of data and visual counter-
    part fits what kind of algorithm.

    The previous 2 phases(Feature Engineering and Selection) are also part of EDA , because based on how my target and feature variables look like, some algorithms are automatically disqualified. 
    An example would be linear regression, the output data is binary 
    and so LR would be the last algorithm on my mind.

    To Clearly see the patterns we want our algorithm to learn, a multivarient analysis must take place, i have used a pairplot(from seaborn):
    
    '''
    st.write(eda_desc)
    st.subheader('Pair Plot')
    st.image(pair)
    st.subheader('Patterns we can see:')
    patterns = '''
    1.The 2 class in relation to the target variable have thresholds
    in which their frequencies diminish and the other class begins to dominate- For most of the plots they are linearly seperable.
    
    2.Data points of point class 0(negative) are more dense and clustered in comparison to class 1 data points.
    class 1 data points are more dispersed.
    '''
    st.write(patterns)
    st.subheader('Visualizing the top 3 variables and infering models:')
    pair = Image.open('Production\pairplot3.png')
    st.image(pair)
    st.subheader('More Analysis')
    st.write('This is a good time to remember our objective for this model, we want a model that will value recall over precision. This just means we don\'t mind false positives , as long as we are not misclassifying  true positives as false negative.')
    st.write('A good start would be Logistic Regression, it is a binary classification algorithm that is worth considering: ')
    st.subheader('1. Logistic Regression:')
    logreg = '''
    As we know logistic regression is a binary classification algorithm that is based on an entropy loss function.
    This is good if our data is split out well, and in this case we have a overlapping 'boundery' and this could pose a problem especially if 
    we are talking entropy because that region(the over lapping data points) has high entropy , meaning the infomation is not clear or is filled with noise.
    '''
    st.write(logreg)
    
    st.subheader('2. Perceptron')
    percp = '''
    The Perceptron(Neuron) is a binary classification algorithm that will start brodening how we look at our problem and how a bias is important in problems like these. Because of how our dataset looks like , an option we have is to have a higher tolarence to false positives in an attempt to catch out all the true positives-better recall score.
    '''
    st.write(percp)
    st.subheader('3. Naive Bayes(Gaussian)')
    nb = '''
    Naive Bayes would be another good choice based on our dataset, using probability and treating every feature as independent , Gaussian naive bayes could have the potential to quiet down the noise around the boundery.
    The results will be highly anticipated.
    
    '''
    st.write(nb)
    st.subheader('4. Decision Trees ')
    dt = '''
    The best for last(based on our data), i think Decision trees will be the best performing algorithm out of the pack. The thresolding nature of decision trees on our feature variables will allow for a more balanced bias, but still setting strict rules(thresholds)
    
    '''
    st.write(dt)
    

    #fig = px.scatter(reduced_frame,x='V17',y='V12',color='Class')
    #st.write(fig)
    