# Implementation-of-Linear-Regression-Using-Gradient-Descent

Program to implement the linear regression using gradient descent.
Developed by: M.syed raashid husain
RegisterNumber: 25009038 
```
# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start

#### Step 2: Import Libraries
#### Import numpy, pandas, and StandardScaler from sklearn.

#### Step 3: Load Dataset
#### Read the CSV file using pandas.read_csv().
#### Read the CSV file using pandas.read_csv().
#### X → Independent variables (all columns except the last).
#### y → Dependent variable (last column).

#### Step 4: Preprocess Data
#### Standardize the features (X) using StandardScaler to improve convergence.
#### Add a bias column (column of ones) to X for the intercept term.

#### Step 5: Initialize Parameters
#### Set all model parameters θ (theta) to zero.
#### Choose a learning rate (α) and number of iterations.

#### Step 6: Hypothesis Function
<img width="595" height="76" alt="image" src="https://github.com/user-attachments/assets/5a4d5814-d040-40a2-a35b-364d840a722f" />

#### Where:
#### X is the feature matrix (with bias column).
#### θ is the parameter vector.

#### Step 7: Cost Function (Mean Squared Error)
![WhatsApp Image 2025-09-13 at 14 37 25_2eab0dec](https://github.com/user-attachments/assets/47fbb6d6-ebd3-4b56-98bd-d3bafbe8de8c)

#### Step 8: Gradient Descent Update Rule
#### Repeat for the chosen number of iterations:
<img width="372" height="50" alt="image" src="https://github.com/user-attachments/assets/9ddb908a-4188-4dcd-8757-397368f17238" />

#### Step 9: Model Training
#### 1.Run gradient descent loop until parameters converge (or max iterations reached).
#### 2.Final learned values of θ represent the trained model.

#### Step 10: Prediction
####     For new input data:
####        1.Scale the features using the same scaler as training.
####        2.Add the bias term.
####        3.Compute:
<img width="188" height="57" alt="image" src="https://github.com/user-attachments/assets/45e3cb48-690c-4ea6-982a-453d8aa66ba8" />

#### Step 11: Output
####    Print learned parameters.
####    Print predicted output for new test data.

#### Step 12: End
## Program:
python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression (X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors= (predictions-y).reshape(-1,1)
    theta -learning_rate*(1/len(X1))*X.T.dot(errors)

    return theta

data=pd.read_csv(r"C:\Introduction to Machine Learning\UNIT 1\Gradient Descent\DATASET-20250226\50_Startups.csv")
data.head(11)

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

print("X =",X)

print("X1_Scaled =",X1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")

```

## Output:
<img width="644" height="461" alt="Screenshot 2025-09-13 141913" src="https://github.com/user-attachments/assets/fbf23537-2a4c-4316-854e-737cfa18684f" />

<img width="507" height="824" alt="Screenshot 2025-09-13 141655" src="https://github.com/user-attachments/assets/96db39a5-62a9-47f9-84e5-e33e3010475e" />

<img width="482" height="212" alt="Screenshot 2025-09-13 141716" src="https://github.com/user-attachments/assets/e24ce494-9aea-4e9b-aac1-0b1f678fdb81" />

<img width="583" height="792" alt="Screenshot 2025-09-13 141742" src="https://github.com/user-attachments/assets/b6b97366-ee7f-4341-ba4d-6c8186e76f9c" />

<img width="485" height="252" alt="Screenshot 2025-09-13 141828" src="https://github.com/user-attachments/assets/d7acda7b-cf46-4854-aaca-c13fd6228211" />

<img width="719" height="44" alt="Screenshot 2025-09-13 141838" src="https://github.com/user-attachments/assets/9a884630-123e-47b9-a442-dd6641c5f0fa" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
