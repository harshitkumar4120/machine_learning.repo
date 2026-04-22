# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Step 2: Sample Dataset (Position vs Salary)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([10000, 20000, 30000, 40000, 50000, 60000])

# Step 3: Model create (RBF kernel)
model = SVR(kernel='rbf')
model.fit(X, y)

# Step 4: Prediction
X_test = np.array([[3.5]])
y_pred = model.predict(X_test)

print("Predicted value for 3.5:", y_pred)

# Step 5: Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("SVR Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()