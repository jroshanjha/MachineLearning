# MachineLearning <br>
# This is First Machine Learning Project <br>
import numpy as np <br>
import pandas as pd  <br>
from pandas.plotting import scatter_matrix <br>
import seaborn  <br>
import matplotlib.pyplot as plt <br>
import scipy <br>
from sklearn import model_selection <br>
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,LinearRegression <br>
from sklearn.model_selection import cross_val_score,KFold,train_test_split <br>
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix,accuracy_score,mean_absolute_error,R2_score <br>
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor <br>
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,kneighbors_graph,KNeighborsTransformer <br> 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis <br>
from sklearn import *  <br>
from sklearn.naive_bayes import GaussianNB  <br>
from sklearn.neural_network import * <br>
from sklearn.svm import SVC  <br>
from sklearn.preprocessing.StandardScaler  <br>
%matplotlib inline  <br>

