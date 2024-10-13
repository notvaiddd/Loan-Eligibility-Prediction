
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    'income': [35000, 40000, 50000, 25000, 30000, 60000, 32000],
    'credit_score': [700, 680, 720, 620, 640, 750, 690],
    'loan_amount': [100000, 200000, 150000, 80000, 120000, 250000, 130000],
    'employment_status': [1, 1, 1, 0, 1, 1, 1],  
    'eligibility': [1, 1, 1, 0, 0, 1, 1]  
}


df = pd.DataFrame(data)


X = df[['income', 'credit_score', 'loan_amount', 'employment_status']]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")


i = int(input("Enter your monthly income (₹): "))  
c = int(input("Enter your credit score: "))         
l = int(input("Enter the loan amount you're requesting (₹): "))  
e = int(input("Are you employed? (1 for yes, 0 for no): "))  


user_data = np.array([[i, c, l, e]])  
prediction = model.predict(user_data)


if prediction == 1:
    print("You are eligible for the loan.")
else:
    print("You are not eligible for the loan.")
