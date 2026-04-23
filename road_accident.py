import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Dataset
# Features typically include: Weather_conditions, Road_surface_type, Light_conditions, Speed_limit, etc.
df = pd.read_csv('road_accidents.csv')

# 2. Data Cleaning
df.dropna(inplace=True) # Removing missing values

# 3. Preprocessing (Encoding categorical variables)
le = LabelEncoder()
categorical_features = ['Weather_conditions', 'Road_surface_type', 'Light_conditions', 'Type_of_collision']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# 4. Feature Selection
X = df.drop(columns=['Accident_severity']) # Target variable
y = df['Accident_severity']

# 5. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Prediction and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

