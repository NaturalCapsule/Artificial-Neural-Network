#Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import Sequential # type: ignore
import tensorflow as tf

# Loading the dataset and initilazing the x and y (feature and target)
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# LabelEncode the gender column because it has binary values
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# Applying OneHotEncode to the feature variable to get rid of the categorical values and converting them into numerical values
ct = ColumnTransformer([('encoder',OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)
x = np.asarray(x).astype(np.float32)

# Splitting the feature and target into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state = 0)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

# Scaling the features because when we use an ANN model it needs to get scaled to get better results
sc = StandardScaler()
x_train, x_test = sc.fit_transform(x_train), sc.transform(x_test)

# Creating 3 hidden layer and one output layer because the prediction we wanna make is classification and not regression
ann = Sequential()
ann.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# Using ann.predict method to predict the test set (which is x_test)
y_ann_pred = ann.predict(x_test)

# Convert probabilities to binary predictions
y_ann_pred_binary = (y_ann_pred >= 0.5).astype(int)

# Evaluating the model
print(accuracy_score(y_test, y_ann_pred_binary))
print(confusion_matrix(y_test, y_ann_pred_binary))