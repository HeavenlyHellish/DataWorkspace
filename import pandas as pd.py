import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Genshin Impact Survey Results.csv", skipinitialspace=True)

# Drop unnamed index column if present
if df.columns[0] == "" or "Unnamed" in df.columns[0]:
    df.drop(columns=df.columns[0], inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ["When did you start playing Genshin Impact", "How much have you spent", "Gender"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop(columns=["How much have you spent"])
y = df["How much have you spent"]

# Split into 70% training, 20% validation, 10% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Save to CSV
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
