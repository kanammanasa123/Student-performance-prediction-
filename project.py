import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'hours_studied': [1,2,3,4,5,6,7,8],
    'attendance': [50,60,65,70,75,80,85,90],
    'passed': [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Input & Output
X = df[['hours_studied', 'attendance']]
y = df['passed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
result = model.predict([[5, 75]])

print("Prediction (1=Pass, 0=Fail):", result[0])