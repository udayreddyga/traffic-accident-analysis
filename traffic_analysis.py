import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("traffic_accidents.csv")

label_enc = LabelEncoder()
for col in ['Weather_Condition', 'Road_Type', 'Vehicle_Type', 'Light_Condition', 'Traffic_Density']:
    data[col] = label_enc.fit_transform(data[col])

X = data[['Weather_Condition', 'Road_Type', 'Vehicle_Type', 'Light_Condition',
          'Speed_Limit', 'Traffic_Density']]
y = LabelEncoder().fit_transform(data['Severity'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
plt.show()
