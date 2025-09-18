import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and drop missing rows
frame = pd.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True).dropna()

# Encode categorical features
for col in ['Gender', 'When did you start playing Genshin Impact']:
    frame[col] = LabelEncoder().fit_transform(frame[col])

# Convert spending to numeric midpoints
spending_map = {
    "Nothing": 0, "Less than $50": 25, "$50 - $100": 75, "$100 - $500": 300,
    "$500 - $1,000": 750, "$1,000 - $5,000": 3000, "Over $5,000": 5000
}
y = frame['How much have you spent'].replace(spending_map)

# Add interaction term and Age squared
frame['Age_x_Start'] = frame['Age'] * frame['When did you start playing Genshin Impact']
frame['Age_squared'] = frame['Age'] ** 2

# Features and scaling
x = frame[['Age', 'When did you start playing Genshin Impact', 'Gender', 'Age_x_Start', 'Age_squared']]
x_scaled = StandardScaler().fit_transform(x)

# Train-test split
x_train, _, y_train, _ = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(x_train, y_train)

# Training R^2
print(f"Training R^2: {model.score(x_train, y_train):.3f}")

# Scatter plot with jitter
plt.figure(figsize=(10, 6))
plt.scatter(frame['Age'], y + np.random.uniform(-0.1, 0.1, len(frame)), color='blue', alpha=0.7, edgecolors='k')
plt.xlabel("Age")
plt.ylabel("How much have you spent")
plt.title("Age vs. Spending")
plt.tight_layout()
plt.show()
