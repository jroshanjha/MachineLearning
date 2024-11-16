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
import sys<br>
import sklearn<br>
import matplotlib<br>
%matplotlib inline  <br>
# z = (x - u) / s



print('Python: {}'.format(sys.version))
print('Scipy: {}'.format(scipy.__version__))
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Sklearn: {}'.format(sklearn.__version__))



# For Random Forest Classifier (used for classification tasks)
from sklearn.ensemble import RandomForestClassifier

# For Random Forest Regressor (used for regression tasks)
from sklearn.ensemble import RandomForestRegressor


# For classification
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(x_train, y_train)

# For regression
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(x_train, y_train)



#spot check algorithms
models=[]
#models.append(('LinearRegression:',LinearRegression()))
models.append(('LR:',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA:',LinearDiscriminantAnalysis()))
models.append(('KNN:',KNeighborsClassifier()))
models.append(('DTC(CART:)',DecisionTreeClassifier()))
#models.append(('DTR:',DecisionTreeRegressor()))
models.append(('RFC: ',RandomForestClassifier(n_estimators=100)))
# models.append(('RFR: ',RandomForestRegressor(n_estimators=100)))
models.append(('NB:',GaussianNB()))
models.append(('SVC',SVC(gamma='auto')))

results =[]
names = []

for name,model in models:
    kfold = KFold(n_splits = 10, random_state = seed,shuffle=True)
    
    cv_results = cross_val_score(model,x_train,y_train,cv= kfold,scoring =scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)
#End for loop 


# comparing Algorithms and select the best model 
fig = plt.figure()
fig.suptitle('Algoriths Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results,showmeans=True)
ax.set_xticklabels(names)
plt.show()



# Missing Value:- mean, modian and mode for categorical 
SimpleImputer(strategy ='mean')
fillna(method='bfill,fill',axis=0/1)
select_dtypes(include='float64').columns
OneHotEncoding -> categorical to Numerical 
OneHotEncoding(drop='first')
pd.get_dummies(drop='firsst')

LabelEncoding -> for nomial 
OrdinalEncoding -> for ordinal
OrdinalEncoding(categories =data)
pd.Categorical(df['col']).codes

IQR = Q3-Q1 
min_val = Q1- 1.5(IQR)
max_val = Q3 + 1.5(IQR)

Z-score 
min_val = x(mean)-3 * x(std) 
max_val = x(mean) + 3 * x(std)

z = x-x(mean)/x(std) 

# Feature Scalling:- 
StandardScaler->Standardization -> outlier have not remove but magniuted changed
x(new) = x-x(mean)/x(std)

# Normalization:- 
MinMaxScaler   
x(new) = x-min(x)/max(x)-min(x)

#Function Transformer
A FunctionTransformer forwards its X (and optionally y) arguments to a user-defined function or function object and returns the result of this function. This is useful for stateless transformations such as taking the log of frequencies, doing custom scaling, etc.

FunctionTransformer(func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None)
FunctionTransformer(func=np.log1p)
