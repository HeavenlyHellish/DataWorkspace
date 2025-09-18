import pandas as pan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
frame = pan.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True)
frame = frame.dropna(axis=0, inplace=False)

# Encode categorical features
frame['Gender'] = LabelEncoder().fit_transform(frame['Gender'])
frame['When did you start playing Genshin Impact'] = LabelEncoder().fit_transform(
    frame['When did you start playing Genshin Impact']
)

# Define custom spending order
spending_order = [
    "Nothing",
    "Less than $50",
    "$50 - $100",
    "$100 - $500",
    "$500 - $1,000",
    "$1,000 - $5,000",
    "Over $5,000"
]

# Convert spending column to ordered categorical
frame['How much have you spent'] = pan.Categorical(
    frame['How much have you spent'],
    categories=spending_order,
    ordered=True
).codes

# Define features and target
X = frame[['Age', 'Gender', 'When did you start playing Genshin Impact']]
y = frame['How much have you spent']

# Split into training (70%), validation (20%), and testing (10%)
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

# Evaluate accuracy
train_acc = rf.score(x_train, y_train)
val_acc = rf.score(x_val, y_val)
test_acc = rf.score(x_test, y_test)

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {val_acc:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}")

# Get feature importances
importances = rf.feature_importances_
print("\nFeature importances (Random Forest):")
for feature, score in zip(X.columns, importances):
    print(f"{feature}: {score:.3f}")

# Plot feature importances
plt.figure(figsize=(8, 5))
plt.barh(X.columns, importances, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
