import pandas as pan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
frame = pan.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True)

# Drop rows with missing data
frame = frame.dropna(axis=0, inplace=False)

# Encode features
frame['Gender'] = LabelEncoder().fit_transform(frame['Gender'])
frame['When did you start playing Genshin Impact'] = LabelEncoder().fit_transform(frame['When did you start playing Genshin Impact'])

# Encode target
frame['How much have you spent'] = LabelEncoder().fit_transform(frame['How much have you spent'])

# Split features and target
x = frame[['Age']]
y = frame['How much have you spent']

# Split into training (70%), validation (20%), and testing (10%)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

# Output the sizes of each split
print("Training set size:", len(x_train))
print("Validation set size:", len(x_val))
print("Testing set size:", len(x_test))

# Calculate R^2 Score
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(x_train, y_train)

r_squared = model.score(x_train, y_train)
print("R^2:", r_squared)

# Create a graph

plt.scatter(x, y, color='blue')

plt.xlabel("Age")  
plt.ylabel("How much have you spent (encoded)")
plt.title("Age vs. Spending")
savefig = "age_vs_spending.png"