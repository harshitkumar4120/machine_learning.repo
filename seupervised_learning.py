import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (Area of house)
X = np.array([[500], [800], [1000], [1200], [1500]])

# Output labels (House price)
y = np.array([100000, 150000, 200000, 230000, 300000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for new house
new_area = np.array([[1100]])
prediction = model.predict(new_area)

print("Predicted House Price:", prediction)