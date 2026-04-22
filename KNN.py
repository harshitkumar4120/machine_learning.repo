# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Sample Dataset (Height vs Weight → Fit/Unfit)
data = {
    'Height': [150, 155, 160, 165, 170, 175, 180, 185],
    'Weight': [50, 55, 60, 65, 70, 75, 80, 85],
    'Result': [0, 0, 0, 1, 1, 1, 1, 1]   # 0 = Unfit, 1 = Fit
}

df = pd.DataFrame(data)

# Step 3: Split data
X = df[['Height', 'Weight']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model create (K = 3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Predict new data
new_data = [[168, 68]]
result = model.predict(new_data)
print("Prediction for (168cm, 68kg):", result)