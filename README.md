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
