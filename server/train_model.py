import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

penguins = sns.load_dataset('penguins')

# Drop rows with missing values
penguins = penguins.dropna()
print(penguins.head())

# Split the data into features and target
X = penguins.drop('species', axis=1)  # Drop one species for binary classification
y = penguins['species']

# Convert categorical variables to numerical
X = pd.get_dummies(X)
# y = pd.get_dummies(y)

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model as a pickle file
with open('./server/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)