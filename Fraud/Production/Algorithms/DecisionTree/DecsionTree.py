
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def node_estimator(df,alpha=0.05):
    return np.round(len(df)*alpha)  


class Node:
    def __init__(self,noise,target_varable,minimum_info=0.05) :
        self.parent = True
        self.leaf = False
        self.noise = noise
        self.target_variable = target_varable
        self.minimum_info = minimum_info
        self.minimum_points = 10
        self.label = ''
        self.left =None
        self.is_left =False
        self.is_right =False

        if self.parent == True:
            self.left_trail = []
            self.right_trail = []
        self.right =None
        self.boundary =None 
        self.previous_threshold = None
        self.impurity = self.confusion(noise[self.target_variable])
        self.noise.attrs['impurity'] = self.impurity
        self.noise.attrs['Parentnode'] = True

        if self.impurity > 0 and self.leaf == False :
            learning = self.split(noise) #if 0 no other feature can improve(lessen) our entropy
            
            if learning != 0 : #triggered when we can improve our entropy with a feature
                self.boundary = learning[5]
                self.right = learning[3]
                self.left = learning[1]

                leftnodedata = learning[0].drop[4] 
                leftnode = Node(leftnodedata,self.target_varable)
                leftnode.label = self.left
                leftnode.parent = False
                leftnode.is_left = True
                leftnode.is_right = False
                previous_data_left =  {'split':[self.boundary,learning[3]],'direction':'left','splitlabel':self.left}
                self.left_trail.insert(-1,previous_data_left)
                leftnode.noise.attrs[f'previous_data'] = previous_data_left
                leftnode.previous_threshold = self.boundary

                rightnodedata =learning[2].drop[4]
                rightnode = Node(rightnodedata,self.target_varable)
                rightnode.label = self.right
                rightnode.parent = False
                rightnode.is_right = True
                rightnode.is_left = False
                previous_data_right =  {'split':[self.boundary,learning[3]],'direction':'right','splitlabel':self.right}
                self.right_trail.insert(-1,previous_data_right)
                rightnode.noise.attrs[f'previous_data'] = previous_data_right
                rightnode.previous_threshold = self.boundary

            if learning == 0:
                #there's no more feature that can lessen our entropy, so we collect the mode of the node and set the node to a leaf.
                self.leaf = True
                self.parent = False
                self.label = self.noise[self.target_variable].mode()

                if self.is_left == True:
                    self.left_trail.index(-1,{'End':True,'label':self.label})
                elif self.is_right == True:
                    self.right_trail.index(-1,{'End':True,'label':self.label})


        elif self.impurity == 0:
            self.leaf = True
            self.label = self.noise[self.target_variable].mode()
            if self.is_left == True:
                    self.left_trail.index(-1,{'End':True,'label':self.label})
            elif self.is_right == True:
                    self.right_trail.index(-1,{'End':True,'label':self.label})
        
        

 
    def trail(self):
      return [self.right_trail,self.left_trail]
    def confusion(self,noise):
        classes = noise.value_counts()
        
        e = 0
        
        for _class in classes:
            
            prop = _class/classes.sum()
            e+=prop*(1-prop)*2
        
        return e

    def split(self,noise):
        feature_data = pd.DataFrame({'feature':[],'Boundery':[],'maxinfo':[],'entropy':[]})
        
        for feature_name,feature_vector in noise.iteritems():

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
                index_data.append(frame)
            max_info = self.impurity - index_data.min()
            if max_info >= self.minimum_info:
                feature_data.loc[len(feature_data.index)] = [feature_name,index_data.idxmin() , max_info, index_data.min()]
                right_noise = noise[noise[feature_name > index_data.idxmin() ]]
                left_noise = noise[noise[feature_name < index_data.idxmin() ]]
                left_mode = left_noise[self.target_variable].mode()
                right_mode = right_noise[self.target_variable].mode()
                return [left_noise,left_mode,right_noise,right_mode,feature_name,index_data.idxmin()]
        return 0 

            
