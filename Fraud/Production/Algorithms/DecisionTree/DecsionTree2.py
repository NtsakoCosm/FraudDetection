import streamlit as st
from PIL import Image


def DecisionTrees():
    st.header('Decision Trees')
    dt = '''
    Decision Trees are best described as thresolding of variables in which the thresold results in the lowest entropy,
    this is then done recursively until certain constraints are met that stop the Tree from growing.
    '''
    st.write(dt)
    st.subheader('The Math :')
    st.subheader('1. Gini Entropy')
    gini_entropy = Image.open('Fraud/Production/Algorithms/DecisionTree/gini_and_entropy.png')
    gini_exp = '''
    The ways in which one can measure the uncertainty/ information are multiple, but very smiliar , so we'll list them out and pick the most concise one.

    Pure Entropy is the equation to the left, with the right being a more compact version of the third picture p*(1-p) 
    '''
    st.write(gini_exp)
    st.image(gini_entropy)
    gini_pure = Image.open('Fraud/Production/Algorithms/DecisionTree/Gini_pure.png')
    st.image(gini_pure,caption='The widely used formula for the gini entropy ')
    st.subheader('What do all these equations mean?')
    st.write('The gini is just one aspect needed for a Decision Tree')
    st.write('it tell us how uncertain the split our algorithm made.')
    st.write('This is determined by the gini formula then after multipled but a weighted average as to filter out nodes with larger data points and ones with small data points')
    st.write('if the entropy is large, but the data is small, then the entropy won\'t effect our decision making')
    st.subheader('The other conditions')
    st.subheader('1. Minimum size of node:')
    condition ='''
    The minimum size of the node is a way of avoiding overfitting the model.
    it is one of the primary conditions when deciding if a split is validated.
    '''
    st.write(condition)
    st.subheader('2. Infomation Gain')
    ig = '''
    This happends when one node is split into 2. The previous node's entropy is subtracted from the current node's entropy.
    if the infomation grain meets the thresold set by the ML engineer then the split is validated.
    '''
    st.write(ig)
    st.subheader('3. Impurity')
    impurity = '''
    What if the split is not validated?
    We measure the mode of the node then set that as the label leaf node.
    impurity is entropy , but during the process(Computation) it has multiple uses.
    if the node is split to a perfect 0 entropy and the node meets the minimum size ,the node is marked as a leaf the data points' label is set as the node's label, if not, the same thing is applied to the current node-the one that failed to split.
    
    '''
    st.write(impurity)
    st.subheader('The full process:')
    process = '''
    The parent node's entropy is calculated.
    features are iterrated through to find a threshold(continuous data), or label(Categorical).
    The first feature(parent node) is marked as a property to the node,the thresolds are then used as the binary split( yes or no - Categorical/ threshold => bigger or equal to >= or smaller < - continuous)
    the nodes there after do the same thing , but this time with a different feature ,if that feature gives us enough infomation gain and the split puts enough data points to meet the minimum size of the node. If the subsequent split has a small entropy and no further split has enough info, the node takes the label of the mode of the node label. As we can see the process is cyclical,binary and scored based on entropy that is calculated through infomation gain(a mouthful i know.I can't believe I used to think Decision Trees were just if statements, boy I was wrong).

    The code i wrote is not fully functioning and i need to finish up some bugs, i didn't meet my deadline for this project so what i want to do is fix up the code as the days progress then i'll update it here if you see where my errors are please don't be afraid to contact me.
    '''
    st.write(process)
    dt2 = '''
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

fraud_df = pd.read_csv('/content/drive/MyDrive/Fraud/creditcard.csv')
feautures = fraud_df.drop('Class', axis=1)
target = fraud_df['Class']

x_train, x_test, y_train, y_test = train_test_split(feautures,target, test_size=0.25,random_state=42)  
train = pd.concat([x_train,y_train],axis=1)
test =  pd.concat([x_test,y_test],axis=1)

def node_estimator(df,alpha=0.0005):
    return np.round(len(df)*alpha)  

estimate = node_estimator(fraud_df)
trailrightnode = []
trailleftnode = []
parentnode = []
class Node:
    def __init__(self,noise,target_varable,parent,is_left,is_right,minimum_info=0.005) :
        self.parent = parent
        self.leaf = False
        self.noise = noise
        self.target_variable = target_varable
        self.minimum_info = minimum_info
        self.minimum_points = 10
        self.counts = self.noise[self.target_variable].value_counts()
        self.label = ''
        self.left =None
        self.is_left =is_left
        self.is_right =is_right
        self.right =None
        self.boundary =None 
        self.previous_threshold = None
        self.impurity = self.confusion(noise[self.target_variable])
        self.noise.attrs['impurity'] = self.impurity
        print(self.parent)
       
        
        
        
        

        if self.impurity > 0 and self.leaf == False :
            learning = self.split(noise) #if 0 no other feature can improve(lessen) our entropy
            
            if learning != 0 : #triggered when we can improve our entropy with a feature
                
                self.boundary = learning['boundary']
                self.right = learning['rightmode']
                self.left = learning['leftmode']
                

                leftnodedata = learning['leftnoise']
                leftnode = Node(leftnodedata ,'Class',False,True,False)
                leftnode.label = self.left
                leftnode.parent = False
                leftnode.is_left = True
                leftnode.is_right = False
                previous_data_left =  {'split':[self.boundary],'direction':'left','splitlabel':self.left}
                trailleftnode.insert(-1,previous_data_left)
                leftnode.noise.attrs[f'previous_data'] = previous_data_left
                leftnode.previous_threshold = self.boundary

                rightnodedata = learning['rightnoise']
                rightnode = Node(rightnodedata,'Class',False,False,True)
                rightnode.label = self.right
                rightnode.parent = False
                rightnode.is_right = True
                rightnode.is_left = False
                previous_data_right =  {'split':[self.boundary],'direction':'right','splitlabel':self.right}
                trailrightnode.insert(-1,previous_data_right)
                rightnode.noise.attrs[f'previous_data'] = previous_data_right
                rightnode.previous_threshold = self.boundary
                

            if learning == 0 and self.parent== True:
                #there's no more feature that can lessen our entropy, so we collect the mode of the node and set the node to a leaf.
                self.leaf = True
                self.label = self.noise[self.target_variable].mode()
                parentnode.append({'End':True,'label':self.label})
                quit()

            if learning == 0 and self.parent== False:
                self.leaf = True
                self.label = self.noise[self.target_variable].mode()
          
                if self.is_left == True:
                    print('its left')
                    
                    trailleftnode.insert(-1,{'End':True,'label':self.label,'direction':'Left'})
                    quit()
                elif self.is_right == True:
                    trailrightnode.insert(-1,{'End':True,'label':self.label,'direction':'Right'})
                    print('its right')
                    quit()

        elif self.impurity == 0:
            self.leaf = True
            self.label = self.noise[self.target_variable].mode()
            if self.is_left == True:
                    trailleftnode.insert(-1,{'End':True,'label':self.label})
            elif self.is_right == True:
                    trailrightnode.insert(-1,{'End':True,'label':self.label})
        
        

 

    def confusion(self,noise):
        classes = noise.value_counts()
        e = 0
        for _class in classes:
            prop = _class/classes.sum()
            e+=prop*(1-prop)*2
        return e


    def split(self,noise):
        
        
        for feature_name,feature_vector in noise.iteritems():
            if feature_name == self.target_variable:
              continue

            index_data = pd.Series()

            for data_point in feature_vector:
                right = noise[noise[feature_name] > data_point].value_counts()
                right_size = len(right)
                if right_size <= self.minimum_points:
                    
                    continue

                left = noise[noise[feature_name] <= data_point].value_counts()
                left_size = len(left)
                if left_size <= self.minimum_points:
                    
                    continue
                right_entropy = self.confusion(right) * right_size/len(feature_vector)
                left_entropy = self.confusion(left) * left_size/len(feature_vector)
                frame = pd.Series({data_point:(right_entropy+left_entropy)})
                index_data = index_data.append(frame)
            
            #print('minimum: ',index_data.min(),'datapoint',index_data.idxmin())
            #print(index_data)
            
            max_info = self.impurity - index_data.min()
            
            if max_info >= self.minimum_info:
                
                right_noise = noise[noise[feature_name] > index_data.idxmin()]
                left_noise = noise[noise[feature_name] <= index_data.idxmin()]
                left_mode = left_noise[self.target_variable].mode()
                right_mode = right_noise[self.target_variable].mode()

                finished = {'leftnoise':left_noise,'leftmode':left_mode,'rightnoise':right_noise,'rightmode':right_mode,'featurename':feature_name,'boundary':index_data.idxmin()}
                print(self.noise)
                return finished
        return 0 

            
par = Node(pd.concat([df[df['Class']==0].sample(5000),df[df['Class'] ==1].sample(500,replace=True)]) ,'Class',True,False,False)
print('Left')
print(trailleftnode)

print('Right')
print(trailrightnode)

    '''
    st.code(dt2)
    
    
    st.subheader('Model: ')
    st.write('This is the sklearn model visual outpur: ')
    dtfig = Image.open('Fraud/Production/Algorithms/DecisionTree/Dt.png')
    st.image(dtfig)
    st.subheader('Decision Tree Metrics: ')
    st.subheader('1.Precision recall curve: ')
    st.image(Image.open('Fraud/Production/Algorithms/DecisionTree/pre_rec.png'))
    st.image(Image.open('Fraud/Production/Algorithms/DecisionTree/aupr.png'))
    st.write('This visualizes the precision and recall trade off,where the higher the recall ,the lower the precision and visa vera, and in this model we are focued on the recall score.')
    st.write('The reason, by the way, we use AUPRC(Area under precision and recall curve, is because it performs well with unbalanced dataset such as the one i am using.')
    st.write('as we can see a decision tree is the best performing algorithm based on our AUPRC metric')
    
    
    
    
    
    st.image('Fraud/Production/Algorithms/DecisionTree/rocsocore.png')
    st.write('The area under the reciever operating characteristic curve.')
    st.write('This visually displays how a higher True positive rate is coupled with an increase in false postives. In an attempt to classify all true positives, we end up making a compromise because we have an imbalanced dataset, and we care more about the recall score and as such, a higher false postive rate isn\'t the worst thing in the world')
    st.image(Image.open('Fraud/Production/Algorithms/DecisionTree/auc.png'))
    

