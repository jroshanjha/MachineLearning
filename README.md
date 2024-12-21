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
A FunctionTransformer forwards its X (and optionally y) arguments to a user-defined function or function object and returns the result of this function. This is useful for stateless transformations such as taking the log of frequencies, doing custom scaling, etc. <br>

FunctionTransformer(func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None) <br>
FunctionTransformer(func=np.log1p) <br>

#Backward & Forward elimination:- 

from mlxtend.feature_selection import SequentialFeatureSelector <br>
SequentialFeatureSelector(estimator, *, n_features_to_select='auto', tol=None, direction='forward', scoring=None, cv=5, n_jobs=None) <br>

PolynomialFeature(degree=2) <br>

## .....................................
## ....................................


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Preprocessing and Regularization Techniques

# 1. Basic Random Forest with L1/L2 Regularization
rf_l1_params = {
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.001, 0.01]  # Cost Complexity Pruning (built-in regularization)
}

# Create a pipeline with scaling and Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# GridSearchCV for hyperparameter tuning and regularization
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        'classifier__' + k: v for k, v in rf_l1_params.items()
    },
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit GridSearch
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)
print("Best Cross-validated Score:", grid_search.best_score_)

# 2. Ensemble Regularization Techniques
# Combining multiple regularization approaches

# Bagging with different regularization parameters
def create_regularized_models():
    models = [
        RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            max_features=features,
            random_state=42
        )
        for depth in [None, 10, 20]
        for split in [2, 5, 10]
        for leaf in [1, 2]
        for features in ['sqrt', 'log2']
    ]
    return models

# Voting Classifier for ensemble regularization
from sklearn.ensemble import VotingClassifier

# Create an ensemble of regularized models
ensemble_models = create_regularized_models()
voting_classifier = VotingClassifier(
    estimators=[(f'model_{i}', model) for i, model in enumerate(ensemble_models)],
    voting='soft'
)

# Fit and evaluate voting classifier
voting_classifier.fit(X_train, y_train)
voting_score = voting_classifier.score(X_test, y_test)
print("\nEnsemble Voting Classifier Score:", voting_score)

# 3. Feature Selection with Regularization
from sklearn.feature_selection import SelectFromModel

# Feature selection with L1 regularization
selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    ), 
    prefit=False
)

# Create a pipeline with feature selection and classifier
feature_selection_pipeline = Pipeline([
    ('selector', selector),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid search with feature selection
feature_selection_grid = GridSearchCV(
    estimator=feature_selection_pipeline,
    param_grid={
        'selector__max_features': [None, 'sqrt', 'log2'],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    cv=5,
    scoring='accuracy'
)

feature_selection_grid.fit(X_train, y_train)
print("\nFeature Selection Best Score:", feature_selection_grid.best_score_)
print("Feature Selection Best Parameters:", feature_selection_grid.best_params_)

# Bonus: Regularization Performance Comparison
regularization_methods = {
    'Basic Random Forest': RandomForestClassifier(random_state=42),
    'GridSearch Regularized': best_model,
    'Voting Ensemble': voting_classifier,
    'Feature Selection': feature_selection_grid.best_estimator_
}

print("\nRegularization Method Comparison:")
for name, model in regularization_methods.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


## Testing ML model 

li= []
for i in range(len(X_test)):
    #print(X_test.iloc[i].values)
    li.append(model.predict([X_test.iloc[i].values]))

for i in li:
    print(i[0])

