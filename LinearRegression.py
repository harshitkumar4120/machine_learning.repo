from sklearn.linear_model import LinearRegression

X=[[1],[2],[3],[4]]
Y=[2 , 4 , 6 , 8]

model=LinearRegression()

model.fit(X,Y)

print(model.predict([[7]]))