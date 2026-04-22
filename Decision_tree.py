# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Sample Dataset (Age vs Salary → Buy/Not Buy)
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 40],
    'Salary': [15000, 20000, 80000, 90000, 70000, 100000, 18000, 60000],
    'Buy': [0, 0, 1, 1, 1, 1, 0, 1]   # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 3: Split data
X = df[['Age', 'Salary']]
y = df['Buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model create
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Predict new data
new_data = [[30, 40000]]
result = model.predict(new_data)
print("Prediction for (Age=30, Salary=40000):", result)