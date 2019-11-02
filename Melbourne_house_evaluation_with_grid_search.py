#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

###Import dataset
df = pd.read_csv('~\Documents\_Useful_Things\Programming\Data_Sets\Melbourne_housing_FULL.csv')

###Scrubbing Process
#Delete unneeded columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Latitude']
del df['Longitude']
del df['Regionname']
del df['Propertycount']

#Removes rows with missing values
df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)

#Convert non-numeric values to numeric form via one-hot encoding
features_df = pd.get_dummies(df, columns  = ['Suburb', 'CouncilArea', 'Type'])

#Delete "price" column to separate it (a dependant variable) out of our independant variables
del features_df['Price']

#Create X and y arrays
X = features_df.values
y = df['Price'].values

###Split the dataset into test/train set (70/30 split) and shuffle (randomize) rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

###Select Algorithm(Gradient Boosting)
model = ensemble.GradientBoostingRegressor()

#Set the configurations that you wish to test. 
#To minimize processing time, limit num. of variables 
#or experiment on each hyperparameter separately.
param_grid = {
    'n_estimators' : [300, 600],    #range of decision trees to be used.
    'max_depth' : [7,9],            #Determines maximum range of layers for each decision tree.
    'min_samples_split' : [3,4],    #Min range of samples needed  before a new branch is created.
    'min_samples_leaf' : [5,],      #Min range of samples needed to be present in a leaf before a new branch is created.
    'learning_rate' : [0.01, 0.02], #Controls the range of rates at which additional trees influence the overall prediction.
    'max_features' : [0.8, 0.8],    #
    'loss' : ['ls','lad','huber']   #Model's error rate:range of methods to be used
}

#Define grid search. Run with four CPUs in parallel if applicable.
gs_cv = GridSearchCV(model, param_grid, n_jobs = 4)

#Run grid search on training data
gs_cv.fit(X_train, y_train)

#Print optimal hyperparameters
print (gs_cv.best_params_)

###Evaluate results
#Training results
mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print ("Training Set Mean Absolute Error: %.2f" % mse)

#Test results
mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mse)

