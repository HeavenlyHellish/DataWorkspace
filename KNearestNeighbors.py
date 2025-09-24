import pandas as pan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load data
frame = pan.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True)

# Encode categorical features
frame['Gender'] = LabelEncoder().fit_transform(frame['Gender'])
frame['When did you start playing Genshin Impact'] = LabelEncoder().fit_transform(frame['When did you start playing Genshin Impact'])

# Apply spending map
spending_map = {
    "Nothing": 0,
    "Less than $50": 1,
    "$50 - $100": 2,
    "$100 - $500": 3,
    "$500 - $1,000": 4,
    "$1,000 - $5,000": 5,
    "$5000+": 6,
    "Over $5,000": 6,
    '"Over $5,000"': 6
}
frame['How much have you spent'] = frame['How much have you spent'].map(spending_map)

# Drop rows with any NaNs (after mapping)
frame = frame.dropna(axis=0)

# Square spending values
# frame['How much have you spent'] = frame['How much have you spent'] ** 2

# Define features and target
x = frame[['When did you start playing Genshin Impact']]
y = frame['How much have you spent']

# Split data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

# Train KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
score = knn.score(x_train, y_train)
print(f"Score for {k} nearest neighbors: {score}")

# Evaluate on training data
y_train_pred = knn.predict(x_train)
print("Confusion Matrix - Train: ")
print(confusion_matrix(y_train, y_train_pred))

print("Classification Report - Train: ")
print(classification_report(y_train, y_train_pred))
