
import streamlit as st
from PIL import Image


def FeatureEngineering():
    
    st.header('Feature Engineering Phase')
    description = '''
    Feature Engineering is the process of manipulating, transforming and selecting features so that proper analysis can be performed.
    Gaps(missing values) in the data are transformed , filled or dropped.
    '''
    st.write(description)
    st.header('Techniques(Checklist): ')
    st.subheader('1.Imputation')
    imputation = '''
    Imputation is the process of handling human made errors or concerns within our data.

    In part this already happened because of the Principal Component Analysis transformation.
    The labels of the transformation were changed for privacy issues.
    This counts as part of the imputation process.
    '''
    st.write(imputation)
    st.subheader('2. Handling Outliers')
    outliers = '''
        "Outliers" are context dependent and are usually because of human error.
        But in our case , cleaning or removing the outliers within our fraud dataset would be like us deleting information that could potentially help us with false negatives and false positives.
        For example:
    '''
    st.write('The "Outliers" here are the natural part of the actual population, so it would be bad practice to remove because above and beyond we might just have a model that is over fit ')
    st.write(outliers)
    outlierpic = Image.open('Fraud/Production/Outliers.png')
    st.image(outlierpic,caption='The plot used the 2 most correlated features, to illustrate outliers visually. The assumption here is that a line can be seen spliting the 2 classes ')

    
    st.subheader('3.Log Transform')
    logtrans = '''
    Log Transform helps normalize our data when we have an unbalanced dataset like ours, but in this instance log transform will only be applied to the frequency of transactions, as this is the only section(so far) that needs to be normalized.
    '''
    st.write(logtrans)
    
    
    st.warning('Unnormalized')
    st.image(Image.open('Fraud/Production/FeatureEngineering/unnormalized.png'))
    
    st.success('Normalized')
    st.image(Image.open('Fraud/Production/FeatureEngineering/normalized.png'))
    st.subheader('4.Scaling')
    scaling = '''
    We scale features as to make the computations lighter and have a consistent scale. 
    '''
    st.write(scaling)
    st.subheader('1.Normalization:')
    
    st.write('We make the value range 0-1, keep the distribution, but this enhances outliers, and as we stated, outliers in this project will be pivotal')
    norm = Image.open('Production/FeatureEngineering/normalization.png')
    st.image(norm)

    st.subheader('2.Standardization:')
    st.write('This form of scaling , takes into account standard deviation, and will keep ouliers\' effect reduced. The mean is 0 and varience is 1, all the data points are subtracted by their mean and the output divided by the variance')
    standard = Image.open('Production/FeatureEngineering/standardization.png')
    st.image(standard)

    st.header('Before Standardization')
    st.image(Image.open('Fraud/Production/FeatureEngineering/beforestnd.png'))
    st.header('After Standardization')
    
    st.image(Image.open('Fraud/Production/FeatureEngineering/afterstnd.png'))
