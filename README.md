# FraudDetection
Fraud Detection Project 
Framing the Business Problem:

    Credit card transactions are collected and labeled fraud or non fraud.
    My Job is to write an algorithm that 
    decreases the money lost through fraud,
    at the same time write an algorithm that 
    won't worsen the experience had by clients 
    (False Positives)

Framing the Data:

    Fraud Dataset of  rows and columns
    The data has gone through a Principle Component Analysis Transformation


Issues:

	More than 95% of the transactions are non-fraudulant
    so this means that by design the dataset is unbalanced, 
    in which the natural way of measuring accuracy:

 	number_of_correctly_classified_points/number_of_points,
    will not mean anything significant.
	A better solution is to use AUPRC, Area Under Precision and Recall Curve



GAME PLAN:



    The data cleaning aspect was already covered by the data engineers 
    who gave us this dataset.

    Exploratory Data Analysis will lead me to Feature Selection and 
    
    Feature engineering.

    Based on the Experience had here I will have a
    couple of machine learning algorithms i would like
    to test out based on how the data looks and the
    Feature Selection stage.
    The metric that will be used is the one stated above : 
    AUPRC , Area Under Precision and Recall Curve.

    The main thing I am going for is 
    to decide on a algorithm that will give me the lowest recall , 
    as a trade off between the 2, recall makes sense because ,
    by default,
    the dataset is unbalanced so our precision from the start is 
    very high
    (Without a machine learning algorithm), 
    but that doesn't mean we can't make our 
    False Negatives lower, and True Positives Higher.

    

    So that is is, I will be going through Exploratory 
    Data Analysis , that will lead to 
    Feature Selection, leading to Feature Engineering, 
    ending up with me choosing a metric( AUPRC ) 
    that I will test between the machine learning 
    algorithms I deem fitting for the problem.

