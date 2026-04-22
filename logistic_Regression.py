# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Sample Dataset (Hours studied vs Pass/Fail)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Step 3: Split data
X = df[['Hours']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model create & train
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Predict new value
hours = [[3.5]]
result = model.predict(hours)
print("Prediction for 3.5 hours:", result)