# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:41:43 2024

@author: Leigh Marasigan
"""

# Artificial Neural Network Template

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1: Data-Preprocessing

# A. Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Excluding the Row Number, Customer ID, and the Surname
Y = dataset.iloc[:, 13].values

# For the Dataset Information
dataset.info()


# B. Encoding the Categorical Data

# B.1. One-Hot Encoding the "Geography" column to create dummy variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer([('Geography', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')

X = column_transformer.fit_transform(X)

# B.2. Label Encoding the "Gender" Column to convert it to Numeric (Dalawa lang naman ang value kaya pwede na label encoder)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X[:, 4] = label_encoder.fit_transform(X[:, 4])
X = X.astype(float)

# C. Splitting the dataset into the Training Dataset and Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size = 0.2, random_state = 0)

# D. Perform Feature Scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_test_standard = X_test.copy()


X_train_standard = standard_scaler.fit_transform(X_train_standard)
X_test_standard = standard_scaler.transform(X_test_standard)

# Part 2: Building the Artificial Neural Network Model

# A. Import the Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# B. Initialize the ANN
classifier = Sequential()

# C. Adding the Input Layer and the First Hidden Layer
classifier.add(Dense(units = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 12))
classifier.add(Dropout(rate = 0.1))

# D. Adding the Second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# E. Adding the Output Layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))



# Part 3: Training the Artificial Neural Network Model

# A. Compile the ANN Model
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# B. Fitting the ANN Model on the Training Dataset
classifier.fit(X_train_standard, Y_train, batch_size = 10, epochs = 50)



# Part 4: Making Predictions and Evaluating the ANN Model

# A. Predict the Output of the Testing Dataset
Y_predict_probability = classifier.predict(X_test_standard)
#Y_predict_boolean = (Y_predict_probability > 0.5)
#Y_predict = Y_predict_boolean.astype(int)

Y_predict = np.rint(Y_predict_probability) # Ito na gagamitin para di na dadaan ng boolean

# B. To Generate and Plot the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict)

import seaborn as sns
plt.figure(figsize = (10, 7))
sns.heatmap(confusion_matrix, annot = True)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

# C. Computing the Hold-Out Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
print("Hold-Out Accuracy:")
print(accuracy)
print(" ")

# D. Generating the Classification Report
from sklearn.metrics import classification_report
print('Classification Report:')
print(classification_report(Y_test, Y_predict))

# E. Predicting the Output of the Single Observation
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 Years Old
# Tenure: 3 Years
# Balance: $60000
# Number of Products: 2
# With Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: $50000

new_prediction = classifier.predict(standard_scaler.transform(np.array([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
new = np.rint(new_prediction)
print(new)

# The Client will stay at the bank



# Part 5: Perform K-Fold Cross Validation to Assess the ANN Model Performance

# A. To Feature SCale the X Variable Using the StandardScaler
X_standard = X.copy()
X_standard = standard_scaler.fit_transform(X_standard)


# B. Building the ANN Classifier Using KerasClassifier
from scikeras.wrappers import KerasClassifier

def classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 12))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

ann_model = KerasClassifier(model=classifier, batch_size=10, epochs = 50, verbose = 0)

# C. Import the StratifieKFold Class
from sklearn.model_selection import StratifiedKFold

# D. Import the Cross Val Score Class
from sklearn.model_selection import cross_val_score

k_fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=0)

# Try the Following Performance Metrics
    # A. accuracy, "accuracy"
    # B. f1-Score, "f1"
    # C. precision, "precision"
    # D. recall, "recall"
    # E. roc-auc, "roc_auc"
    
# For the Accuracy as Scoring Metrics for the Cross-Validation
accuracies = cross_val_score(estimator=ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy", n_jobs = -1)
accuracies_average = accuracies.mean()
accuracies_standard_deviation = accuracies.std()

print('Accuracies of k-folds:', accuracies)
print(" ")
print("Average of the Accuracies of k-folds:", accuracies_average)
print(" ")
print('Standard Deviation of the Accuracies of k-foldsP:', accuracies_standard_deviation)
print(" ")

# For the F1 as Scoring Metrics for the Cross-Validation
f1 =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'f1', n_jobs = -1))
f1_average = f1.mean()
f1__standard_deviation = f1.std()

print('F1-Score of the k-folds:', f1)
print(" ")
print('Average of F1-Score of the k-folds:', f1_average)
print(" ")
print('Standard Deviation of F1-Score of the k-fold:', f1__standard_deviation)
print(' ')

# For the Precision as Scoring Metrics for the Cross-Validation
precision =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'precision', n_jobs = -1))
precision_average = precision.mean()
precision__standard_deviation = precision.std()

print('Precision of the k-folds:', precision)
print(" ")
print('Average of Precision of the k-folds:', precision_average)
print(" ")
print('Standard Deviation of Precision of the k-fold:', precision__standard_deviation)
print(' ')

# For the Recall as Scoring Metrics for the Cross-Validation
recall =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'recall', n_jobs = -1))
recall_average = recall.mean()
recall__standard_deviation = recall.std()

print('Recall of the k-folds:', recall)
print(" ")
print('Average of Recall of the k-folds:', recall_average)
print(" ")
print('Standard Deviation of Recall of the k-fold:', recall__standard_deviation)
print(' ')

# For the Roc-Auc as Scoring Metrics for the Cross-Validation
rocauc =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y_smote.ravel(), cv = k_fold, scoring = 'roc_auc', n_jobs = -1))
rocauc_average = rocauc.mean()
rocauc__standard_deviation = rocauc.std()

print('ROC-AUC of the k-folds:', rocauc)
print(" ")
print('Average of ROC-AUC of the k-folds:', rocauc_average)
print(" ")
print('Standard Deviation of ROC-AUC of the k-fold:', rocauc__standard_deviation)
print(' ')


# Hold - Out Valdation
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict)
print('Classification Accuracy: %.4f'
      %classification_accuracy)
print('')


# B. For the Classification Error
classification_error = 1 - classification_accuracy
print('Classification Error: %.4f'
      %classification_error)
print('')

# C. For the Sensitivity, Recall Score, Probablity of Detection, True Positive Rate
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict, average = 'weighted')
print('Sensitivity: %.4f'
      %sensitivity)
print('')
# Madaming pwede magenerate, pero for consistency laging weighted ang gagamitin sa average

# D. For the Specificity, True Negative Rate
specificity = TN / (TN+FP)
print('Specificity: %.4f'
      %specificity)
print('')
# Pag imbalanced dataset meaning di equal ang rate ng dalawa ay magiging biased yon, kaya need mo siya isubject agad sa smote para maging balanced siya
# sa SMOTE = di nagdadagdag ng raw samples, pero pinapaanak lang yung ibang samples

# E. For the False Positive Rate
false_positive_rate = 1 - sensitivity
print('False Positive Rate: %.4f'
      %false_positive_rate)
print('')

# F. For the False Negative Rate
false_negative_rate = 1 - specificity
print('False Negative Rate: %.4f'
      %false_negative_rate)
print('')

# G. For the Precision or Positive Predictive Value
# Precision = When the actual postive is predicted, gano kadalas na yung predicted ay laging tama 
from sklearn.metrics import precision_score
precision_score = precision_score(Y_test, Y_predict, average = 'weighted')
print('Precision Score: %.4f'
      %precision_score)
print('')

# H. For the F1-Score
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict, average = 'weighted')
print('F1-Score: %.4f'
      %f1_score)
print('')


# Partr 7: Perform Hyperparameter Tuning to Optimized the ANN

# A. Tune First the Batch Size and Epochs

# A.1, Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier

def classifier_optimization():
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier_optimization.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, verbose = 0)

# A.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# A.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'batch_size': [50, 100, 150, 200, 250],
              'epochs': [10, 50, 100, 150, 200]}
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# A.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# A.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")

# B. Tune Next the Optimizer

# B.1, Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier

def classifier_optimization(optimizer="sgd"):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier_optimization.add(Dropout(rate = 0.1))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier_optimization.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, epochs = 150, batch_size = 50, verbose = 0)

# B.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# B.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'optimizer': ['adam', 'sgd', 'rmsprop', 'adamW', 'adadelta', 'adagrad', 'adamax', 'adafactor', 'aadam', 'ftrl', 'lion', 'Loss Scale Optimizer']}

grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# B.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# B.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")

# C. Tune Next the Optimizer's Learning Rate and Momentum  (Nakadepende kung may Learning Rate and Momentum yung lalabas na optimizer, magbased dun sa link)

# C.1. Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import SGD # Import always yung nahanap na best optimizer sa taas, need lagi iimport, MAGBASED SA LINK KUNG PAANO IIIMPORT AND YUNG DOCUMENTATION NG BAWAT OPTIMIZER

def classifier_optimization(learning_rate, momentum):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    optimizer_setting = SGD(learning_rate = learning_rate, momentum = momentum)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, epochs = 150, batch_size = 50, learning_rate = 0.001, momentum = 0.0, verbose = 0)

# C.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# C.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'learning_rate': [0.001, 0.01, 0.1, 1],
              'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]}
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# C.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# C.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")


# D. Tune Next the Network's Weight Initialization

# D.1. Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.optimizers import SGD # Import always yung nahanap na best optimizer sa taas, need lagi iimport, MAGBASED SA LINK KUNG PAANO IIIMPORT AND YUNG DOCUMENTATION NG BAWAT OPTIMIZER

def classifier_optimization(kernel_initializer):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = kernel_initializer, activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = kernel_initializer, activation = 'relu'))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = kernel_initializer, activation = 'sigmoid'))
    optimizer_setting = SGD(learning_rate = 0.01, momentum = 0.4)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, kernel_initializer = 'golorot_uniform', epochs = 150, batch_size = 50, verbose = 0) # Nawala na learning rate and momentum kasi nilagay na sa optimizer settings

# D.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# D.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform']}
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# D.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# D.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")

# E. Tune Next the Neuron Activation Function

# E.1. Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.optimizers import SGD # Import always yung nahanap na best optimizer sa taas, need lagi iimport, MAGBASED SA LINK KUNG PAANO IIIMPORT AND YUNG DOCUMENTATION NG BAWAT OPTIMIZER

def classifier_optimization(activation1, activation2, activation3):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = 'normal', activation = activation1, input_dim = 12))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = 'normal', activation = activation2))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'normal', activation = activation3))
    optimizer_setting = SGD(learning_rate = 0.01, momentum = 0.4)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, activation1 = 'softmax', activation2 = 'softmax', activation3 = 'softmax', epochs = 150, batch_size = 50, verbose = 0) # Nawala na learning rate and momentum kasi nilagay na sa optimizer settings

# E.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# E.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'activation1': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
              'activation2': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
              'activation3': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}

grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# E.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# E.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")


# F. Tune Next the Dropout Regularization

# F1. Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier


def classifier_optimization(dropout_rate):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = 'normal', activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dropout(rate = dropout_rate))
    classifier_optimization.add(Dense(units = 6, kernel_initializer = 'normal', activation = 'tanh'))
    classifier_optimization.add(Dropout(rate = dropout_rate))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))
    classifier_optimization.add(Dropout(rate = dropout_rate))
    optimizer_setting = SGD(learning_rate = 0.01, momentum = 0.4)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization, dropout_rate = 0.0, epochs = 150, batch_size = 50, verbose = 0) # Nawala na learning rate and momentum kasi nilagay na sa optimizer settings

# F.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# F.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.6, 0.8]} # Pwede pang magdagdag

grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# F.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# F.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")


# G. Tune Next the Number of Neurons in the Hidden Layer

# G1. Build the ANN Model for the Optimization Process
from scikeras.wrappers import KerasClassifier


def classifier_optimization(neurons1, neurons2):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = neurons1, kernel_initializer = 'normal', activation = 'relu', input_dim = 12))
    classifier_optimization.add(Dropout(rate = 0.0))
    classifier_optimization.add(Dense(units = neurons2, kernel_initializer = 'normal', activation = 'tanh'))
    classifier_optimization.add(Dropout(rate = 0.0))
    classifier_optimization.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))
    classifier_optimization.add(Dropout(rate = 0.0))
    optimizer_setting = SGD(learning_rate = 0.01, momentum = 0.4)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model=classifier_optimization,  epochs = 150, batch_size = 50, verbose = 0, neurons1 = 5, neurons2 = 5) # Nawala na learning rate and momentum kasi nilagay na sa optimizer settings

# G.2. To Import The GridSearchCV class
from sklearn.model_selection import GridSearchCV

# G.3. To Set Parameters to be Optimized for the ANN Model
parameters = {'neurons1': [5, 10, 15, 20, 25, 30],
              'neurons2': [5, 10, 15, 20, 25, 30]} # Pwede pang magdagdag

grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = k_fold,
                           n_jobs = -1)

grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# G.4. To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# G.5. To Identify the Best Accuracy and the Best Features
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score:")
print(best_accuracy)
print(" ")
print("Best Parameters:")
print(best_parameters)
print(" ")


# Make Summary of the Best Results

# batch_size = 50
# epochs = 150
# optimizer = SGD
# learning_rate = 0.01
# momentum = 0.4
# kernel_initializer = 'normal'
# activation1 = 'relu'
# activation2 = 'tanh'
# activation3 = 'sigmoid'
# dropout_rate = 0.0
# neurons1 = 5
# neurons2 = 30


# Part 8: Repeat Part 2 to Part 6

# Repeating Part 2: Building the Artificial Neural Network Model

# A. Import the Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# B. Initialize the ANN
classifier_final = Sequential()

# C. Adding the Input Layer and the First Hidden Layer
classifier_final.add(Dense(units = 5, kernel_initializer = 'normal', activation = 'relu', input_dim = 12))
classifier_final.add(Dropout(rate = 0.0))

# D. Adding the Second Hidden Layer
classifier_final.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'tanh'))
classifier_final.add(Dropout(rate = 0.0))

# E. Adding the Output Layer
classifier_final.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))

# Repeating Part 3: Training the Artificial Neural Network Model

# A. Compile the ANN Model
classifier_final.compile(optimizer = 'sgd', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# B. Fitting the ANN Model on the Training Dataset
classifier_final.fit(X_train_standard, Y_train, batch_size = 50, epochs = 150)

print(classifier_final.summary())

# Repeating Part 4: Making Predictions and Evaluating the ANN Model

# A. Predict the Output of the Testing Dataset
Y_predict_probability = classifier_final.predict(X_test_standard)
Y_predict = np.rint(Y_predict_probability) # Ito na gagamitin para di na dadaan ng boolean

# B. To Generate and Plot the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict)

import seaborn as sns
plt.figure(figsize = (10, 7))
sns.heatmap(confusion_matrix, annot = True)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

# C. Computing the Hold-Out Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
print("Hold-Out Accuracy:")
print(accuracy)
print(" ")

# D. Generating the Classification Report
from sklearn.metrics import classification_report
print('Classification Report:')
print(classification_report(Y_test, Y_predict))

# E. Predicting the Output of the Single Observation
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 Years Old
# Tenure: 3 Years
# Balance: $60000
# Number of Products: 2
# With Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: $50000

new_prediction = classifier_final.predict(standard_scaler.transform(np.array([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
new = np.rint(new_prediction)
print(new)

# Repeating Part 5: Perform K-Fold Cross Validation to Assess the ANN Model Performance

# A. To Feature SCale the X Variable Using the StandardScaler
X_standard = X.copy()
X_standard = standard_scaler.fit_transform(X_standard)


# B. Building the ANN Classifier Using KerasClassifier
from scikeras.wrappers import KerasClassifier

def classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'normal', activation = 'relu', input_dim = 12))
    classifier.add(Dropout(rate = 0.0))
    classifier.add(Dense(units = 15, kernel_initializer = 'normal', activation = 'tanh'))
    classifier.add(Dropout(rate = 0.0))
    classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))
    classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

ann_model = KerasClassifier(model=classifier, batch_size=50, epochs = 150, verbose = 0)

# C. Import the StratifieKFold Class
from sklearn.model_selection import StratifiedKFold

# D. Import the Cross Val Score Class
from sklearn.model_selection import cross_val_score

k_fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=0)

# Try the Following Performance Metrics
    # A. accuracy, "accuracy"
    # B. f1-Score, "f1"
    # C. precision, "precision"
    # D. recall, "recall"
    # E. roc-auc, "roc_auc"
    
# For the Accuracy as Scoring Metrics for the Cross-Validation
accuracies = cross_val_score(estimator=ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy", n_jobs = -1)
accuracies_average = accuracies.mean()
accuracies_standard_deviation = accuracies.std()

print('Accuracies of k-folds:', accuracies)
print(" ")
print("Average of the Accuracies of k-folds:", accuracies_average)
print(" ")
print('Standard Deviation of the Accuracies of k-foldsP:', accuracies_standard_deviation)
print(" ")

# For the F1 as Scoring Metrics for the Cross-Validation
f1 =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'f1', n_jobs = -1))
f1_average = f1.mean()
f1__standard_deviation = f1.std()

print('F1-Score of the k-folds:', f1)
print(" ")
print('Average of F1-Score of the k-folds:', f1_average)
print(" ")
print('Standard Deviation of F1-Score of the k-fold:', f1__standard_deviation)
print(' ')

# For the Precision as Scoring Metrics for the Cross-Validation
precision =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'precision', n_jobs = -1))
precision_average = precision.mean()
precision__standard_deviation = precision.std()

print('Precision of the k-folds:', precision)
print(" ")
print('Average of Precision of the k-folds:', precision_average)
print(" ")
print('Standard Deviation of Precision of the k-fold:', precision__standard_deviation)
print(' ')

# For the Recall as Scoring Metrics for the Cross-Validation
recall =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = 'recall', n_jobs = -1))
recall_average = recall.mean()
recall__standard_deviation = recall.std()

print('Recall of the k-folds:', recall)
print(" ")
print('Average of Recall of the k-folds:', recall_average)
print(" ")
print('Standard Deviation of Recall of the k-fold:', recall__standard_deviation)
print(' ')

# For the Roc-Auc as Scoring Metrics for the Cross-Validation
rocauc =  (cross_val_score(estimator = ann_model, X = X_standard, y = Y_smote.ravel(), cv = k_fold, scoring = 'roc_auc', n_jobs = -1))
rocauc_average = rocauc.mean()
rocauc__standard_deviation = rocauc.std()

print('ROC-AUC of the k-folds:', rocauc)
print(" ")
print('Average of ROC-AUC of the k-folds:', rocauc_average)
print(" ")
print('Standard Deviation of ROC-AUC of the k-fold:', rocauc__standard_deviation)
print(' ')

# Repeating Part 6: Hold-Out Validation
# Hold - Out Valdation
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict)
print('Classification Accuracy: %.4f'
      %classification_accuracy)
print('')


# B. For the Classification Error
classification_error = 1 - classification_accuracy
print('Classification Error: %.4f'
      %classification_error)
print('')

# C. For the Sensitivity, Recall Score, Probablity of Detection, True Positive Rate
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict, average = 'weighted')
print('Sensitivity: %.4f'
      %sensitivity)
print('')

# D. For the Specificity, True Negative Rate
#specificity = TN / (TN+FP)
#print('Specificity: %.4f'
#      %specificity)
#print('')

# E. For the False Positive Rate
false_positive_rate = 1 - sensitivity
print('False Positive Rate: %.4f'
      %false_positive_rate)
print('')

# F. For the False Negative Rate
false_negative_rate = 1 - specificity
print('False Negative Rate: %.4f'
      %false_negative_rate)
print('')

# G. For the Precision or Positive Predictive Value
# Precision = When the actual postive is predicted, gano kadalas na yung predicted ay laging tama 
from sklearn.metrics import precision_score
precision_score = precision_score(Y_test, Y_predict, average = 'weighted')
print('Precision Score: %.4f'
      %precision_score)
print('')

# H. For the F1-Score
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict, average = 'weighted')
print('F1-Score: %.4f'
      %f1_score)
print('')

# Done Template