# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Arivazhagan G R
RegisterNumber:  212223040020
*/
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png

![320173539-d7c7ba51-b0c1-4533-8148-15babd5025cf](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/c4043349-49d1-435d-87c3-26b714a34846)

![320173778-e5eb7eb9-4eed-4f66-abbd-873bf0a98893](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/abe11c2e-4384-48fb-a95d-af1f07e23041)

![320173789-bd45fc1e-dbcd-4df4-a4eb-134424faa703](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/01a559b6-016e-497d-a2bf-dc45910ea186)

![320173795-e8bfe98c-2d56-41bc-a3ba-4c9bae08a395](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/69e10206-3c57-456b-9cdc-b07d7de7daf1)

![320173800-5b76c3c8-c833-423f-b826-bed7db1bb5c2](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/5841aedf-5ffd-4ac6-8838-09f76dd74a24)

![320173805-61f2c3ee-6220-4efa-853f-0af6b8332324](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/61dbc306-a802-4e3d-acbc-5702aad1bc3b)

Accuracy:
0.985


![320173814-441719a0-d664-4283-ae45-c03d741ccd75](https://github.com/ARIVAZHAGAN04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161414455/0a80c8d1-cc82-450d-ae03-bba779f4ea9b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
