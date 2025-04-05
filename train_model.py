from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report 
import joblib 

iris = load_iris() 
X, y = iris.data, iris.target  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
) 

clf = RandomForestClassifier(n_estimators=100, random_state=42) 
clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
report = classification_report(y_test, y_pred, target_names = iris.target_names)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model to a file
joblib.dump(clf, "iris_model.pkl")