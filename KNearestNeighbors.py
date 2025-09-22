# K-NEAREST NEIGHBORS MODEL

import pandas as pan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

frame = pan.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True)
frame = frame.dropna(axis=0, inplace=False)
frame['Gender'] = LabelEncoder().fit_transform(frame['Gender'])
frame['When did you start playing Genshin Impact'] = LabelEncoder().fit_transform(frame['When did you start playing Genshin Impact'])

# Encode the target variable (spending)
spending_encoder = LabelEncoder()
frame['How much have you spent'] = spending_encoder.fit_transform(frame['How much have you spent'])
spending_labels = spending_encoder.classes_

# Encode the features
ohe = OneHotEncoder(sparse_output=False, drop='first')
x = ohe.fit_transform(frame[['When did you start playing Genshin Impact']])

# Define features and target
x = frame[['When did you start playing Genshin Impact']]
y = frame['How much have you spent']

# Split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

# KNN Model Loop

for k in [1, 3, 5, 7, 11, 19, 65]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    score = knn.score(x_train, y_train)
    print(f"Score for {k} nearest neighbors: {score}")


"""
k = 4
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
score = knn.score(x_train, y_train)
print(f"Score for {k} nearest neighbors: {score}")
"""