import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('onlinefraud.csv')

# Handle missing values
data = data.dropna()

# Handle categorical variables
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])
data['nameOrig'] = le.fit_transform(data['nameOrig'])
data['nameDest'] = le.fit_transform(data['nameDest'])



# Select the model
model = RandomForestClassifier()



# Split the data into training and test sets
X = data.drop('isFraud', axis=1)

y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

#saving the models ....

joblib.dump(model, 'online-fraud-model.joblib')



# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
