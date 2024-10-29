import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data= pd.read_excel('Thyroid_Diff_OneHotEncoded.xlsx')

x=data.drop(columns=['Recurred_No','Recurred_Yes'])
y=data['Recurred_Yes']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

rf_model=RandomForestClassifier(random_state=42)

rf_model.fit(x_train,y_train)

y_pred =rf_model.predict(x_test)
 
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)

print("Classification Report: ")
print(classification_report(y_test,y_pred))