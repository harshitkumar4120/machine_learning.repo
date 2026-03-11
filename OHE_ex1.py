from sklearn.preprocessing import OneHotEncoder
import numpy as np

# categorical data
data = np.array([['Red'], ['Green'], ['Blue'], ['Red']])

# create encoder
encoder = OneHotEncoder()

# fit and transform
encoded = encoder.fit_transform(data)

# convert to array
result = encoded.toarray()

print(result)

