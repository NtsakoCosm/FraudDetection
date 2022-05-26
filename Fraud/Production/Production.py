from EDA import Eda
from FeatureSelection import FeatureSelection
from FeatureEngineering import FeatureEngineering
from Algorithms.Perceptron.Perceptron import perceptron
from Algorithms.NaiveBayes.NaiveBayes import Gauss
from index import index_page
import streamlit as st
from Algorithms.DecisionTree import DecsionTree2 as dt
from Algorithms.LogisticRegression import LogisticRegression



opt = st.sidebar.radio('Choose a Point in the LifeCycle',['Index','Feature Engineering','Feature Selection','EDA','Perceptron','Logistic Regression','Naive Bayes', 'Decision Tree'])


if opt == 'Index':
    index_page()
    pass

elif opt == 'EDA':
    Eda()
    pass
    

elif opt == 'Feature Selection':
    FeatureSelection.FeatureSelection()
    pass

elif opt == 'Feature Engineering':
    FeatureEngineering.FeatureEngineering()
    pass

elif opt == 'Perceptron':
    perceptron()
    pass
    

elif opt == 'Logistic Regression':
    LogisticRegression.LogisticRegProduction()
    pass
    

elif opt == 'Naive Bayes':
    Gauss()
    pass


elif opt == 'Decision Tree':
    dt.DecisionTrees()
    pass