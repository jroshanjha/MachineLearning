# MachineLearning
# This is First Machine Learning Project
import numpy as np
import pandas as pd 
from pandas.plotting import scatter_matrix
import seaborn 
import matplotlib.pyplot as plt
import scipy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,LinearRegression
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix,accuracy_score,mean_absolute_error,R2_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,kneighbors_graph,KNeighborsTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import * 
from sklearn.svm import SVC
from sklearn.preprocessing.StandardScaler
%matplotlib inline

