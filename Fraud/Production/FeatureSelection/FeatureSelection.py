
import numpy as np

import streamlit as st
from PIL import Image



def FeatureSelection():
    st.header('Feature Selection Phase')
    image = Image.open('Fraud/Production/point-biserial-correlation.png')
    synopsis = '''
    Feature Selection involves using metrics to decide 
    which features have the most amount of infomation
    relative to the target variable.

    The dataset has 30 Features and 200 000 Rows,

    Which means:

        -Exploratory Data Analysis will be expensive.
        -Curse of Dimensionality
        -If you have 2 or more Feature variables that are highly correlated,
        one might risk overfitting.
        -Multiple features that don't add infomation, will have
        high cost but low benefit.
    '''
    st.write(synopsis)
    st.image(Image.open('Fraud/Production/FeatureSelection/frame.png'))
    st.header("Techniques:")
    #1
    st.subheader('1. Point biserial correlation')
    st.write(
        '''Point biserial correlation is the most suitable statistical test for this context.
        We are using continuous data for binary classification.
        This test was built Exactly for that 
        ''')
    st.image(image,caption='Point biserial correlation Formula')
    st.write(
        'The Formula Reads: the mean of the first class minus the mean of the second class divided by the standard deviation of the whole dataset multiplied by the square root of the product of the 2 propotions'
    )

    st.write('So here is the code i wrote:')
    
    # Using the test against every other feature, and measuring the coefficients
    
    #Sorting the coefficients
    
    #Chosing the K-best features
    
    pbc = f'''
            coe_values = dict
            # Using the test against every other feature, and measuring the coefficients
            for (colname,coldata) in train.iteritems():
                val = stats.pointbiserialr(coldata,target_train).correlation
                coe_values[colname] = val
            #Sorting the coefficients
            correlation = sorted(coe_values, key=coe_values.get)
            highest_corr = dict
            #Chosing the K-best features
            best = st.number_input('Choose a k')
            kbest = best
            for i in correlation:
                highest_corr[i] = coe_values[i]
                kbest+=1
                if kbest >= kbest:
                    break

'''
    st.code(pbc)
    st.write('Note, i have taken the absoulute value of the coefficients, making it easier to sort through the data.')
    st.image(Image.open('Fraud/Production/FeatureSelection/pbcoe.png'))
    st.write('As we can see we have selected a k, and we can list out the best correlation scored features from our statistical test.')

    st.header('2.Infomation Gain(Entropy) ')
    st.write('Mutual infomation gain is essientially a measure of how much infomation a variable has relative to having no variable at all')
    st.write('What is the entropy in general?, What is the entropy after we split the target variable based on a feature, then we as how much entropy did we lose?, the higher the infomation gain the more valuable the feature')

         
    st.image(Image.open('Fraud/Production/FeatureSelection/infogain.png'))

    st.subheader('The 2 metrics both did a similar job in indentifying infomation dense features. The only deviation would be: V7 and V9, we can add either one to the other and end up with 11 features')

