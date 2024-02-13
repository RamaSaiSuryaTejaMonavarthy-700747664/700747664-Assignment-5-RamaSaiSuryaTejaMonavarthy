import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
glass_data = pd.read_csv('glass.csv')
X = glass_data.drop(["Type"], axis=1) 
y = glass_data["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
naive_bayes_model = GaussianNB() 
naive_bayes_model.fit(X_train, y_train)
y_pred = naive_bayes_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy Score: {:.2f}%".format(accuracy * 100)) 
print("\nClassification Report:\n", report)
