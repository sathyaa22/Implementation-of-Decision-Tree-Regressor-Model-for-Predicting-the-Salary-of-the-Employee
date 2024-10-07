# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import the standard libraries.

Step 3: Upload the dataset and check for any null values using .isnull() function.

Step 4: Import LabelEncoder and encode the dataset.

Step 5: Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

Step 6: Predict the values of arrays.

Step 7: Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

Step 8: Predict the values of array.

Step 9: Apply to new unknown values.

Step 10: End the program.

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: SATHYAA R

RegisterNumber: 212223100052

```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

### Mean Squared Error:

![Screenshot 2024-10-07 110418](https://github.com/user-attachments/assets/720ac9a3-e63e-4960-8611-211b3555db60)

### R2:

![Screenshot 2024-10-07 110437](https://github.com/user-attachments/assets/81db02cf-c25d-46b6-9883-053c76a059a4)

### Prediction:

![Screenshot 2024-10-07 110512](https://github.com/user-attachments/assets/2cdedea0-3456-4546-8572-8c5bf66ca701)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
