# Check Versions
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Working Versions
#Python: 3.5.2 (default, Nov 12 2018, 13:43:14) 
#[GCC 5.4.0 20160609]
#scipy: 0.17.0
#numpy: 1.16.3
#matplotlib: 1.5.1
#pandas: 0.24.2
#sklearn: 0.21.1

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# The idea here is to separate the data that we have into
# two portions.  One portion will be used for "training"
# the algorithm and the other portion will be used for
# "verifying" the accuracy of the model.
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# As a result, X_train and Y_train will be used for training
# whereas X_validataion and Y_validataion will be used for
# verification
# NOTE: The X values are considered input data and Y values are
#   considered output data.  In this case, each entry of X is
#   an array of four measurements of the flower and the
#   corresponding entry of Y is the type of flower.  We're
#   going to train a model (using X_train and Y_train) to use
#   the values in X_validation to predict the value in
#   Y_validataion.
# NOTE: The sytax used to create X and Y is called "slicing."

###############################################
# Time to build, evaluate, and train the models
###############################################

# Some options used in the models.
seed = 9287 # This is an RNG seed; it can be basically anything.
scoring = 'accuracy'

# Building the models.
models = []
# Logistic Regression
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# Linear Discriminant Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))
# K-Nearest Neighbors
models.append(('KNN', KNeighborsClassifier()))
# Classification and Regression Trees
models.append(('CART', DecisionTreeClassifier()))
# Gaussian Naive Bayes
models.append(('NB', GaussianNB()))
# Support Vector Machines
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn

# Evaluating the models.
bestAccuracy = 0
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print("Accuracy Score: " + str(accuracy_score(Y_validation, predictions)))
    print(classification_report(Y_validation, predictions))
    print("**********************************")
    print()
    # At the end of this loop we make sure we pick out the one
    # model that has the best accuracy.
    if cv_results.mean() > bestAccuracy:
        bestModel = model
        bestModelName = name
        bestPredictions = predictions

# Finally, we go through the best model and for each of the
# verification samples we check if the predicted outcome matches
# the actual outcome.
print("Best Model: " + bestModelName)
for i in range(0, len(X_validation)):
    msg = "Verification Sample " + str(i) + " "
    if (bestPredictions[i] == Y_validation[i]):
        msg = msg + "matched expected outcome."
    else:
        msg = msg + "DID NOT MATCH!"
    print(msg)
