import warnings 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score, classification_report 
warnings.filterwarnings("ignore")
glass_data = pd.read_csv('glass.csv')
features = glass_data.drop(['Type'], axis=1) 
target = glass_data["Type"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) 
svm_model = LinearSVC(random_state=42) 
svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100)) 
print("\nClassification Report:\n", report)
