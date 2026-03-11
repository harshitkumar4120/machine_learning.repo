# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Training data (Hours studied vs Marks)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([35, 40, 50, 60, 70])


# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Prediction
hours = np.array([[8]])
predicted_marks = model.predict(hours)

print("Predicted Marks for 6 hours study:", predicted_marks[0])

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression Example")
plt.show()