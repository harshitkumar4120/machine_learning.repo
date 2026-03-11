import numpy as np
from sklearn.linear_model import LinearRegression

# Input features: [Area, Bedrooms, Age]
X = np.array([
    [1000, 2, 5],
    [1500, 3, 10],
    [1800, 4, 7],
    [1200, 2, 6],
    [2000, 4, 3]
])

# Output (House Price)
y = np.array([200000, 300000, 350000, 220000, 400000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict new house price
# Example: Area=1600, Bedrooms=3, Age=5
prediction = model.predict([[1600, 3, 5]])

print("Predicted House Price:", prediction)