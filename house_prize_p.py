import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Area": [1000,1500,1800,2000,2200,2500,2800,3000],
    "Bedrooms": [2,3,3,3,4,4,4,5],
    "Age": [10,5,7,3,2,1,4,2],
    "Price": [300000,400000,420000,500000,520000,600000,650000,700000]
}

df = pd.DataFrame(data)

X = df[["Area","Bedrooms","Age"]]
y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("Predicted Prices:", predictions)

new_house = [[2400,4,3]]  
price = model.predict(new_house)

print("Predicted Price for new house:", price)