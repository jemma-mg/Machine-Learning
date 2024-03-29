# -*- coding: utf-8 -*-
"""ML_workshop.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b3TMluNBUzvhKYkzNKOdTIVVFuFG6FNe
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

"""### Linear Regression"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Mult_Reg_Yield.csv")

df.head()

df.describe()

df.info()

df.isna().sum()

# df.fillna()

plt.boxplot(df["Time"])
plt.title("Box Plot")
plt.show()

X = df.iloc[:,0:2]
X

y = df.Yield
y

sns.pairplot(df)
plt.show()

lr = LinearRegression()

model = lr.fit(X,y)

"""Linear Regression
Yield = B0 + (B1 x Time) + (B2 x Temperature)
"""

model.coef_ 
#to find coefficients from the model
#these are the B1 and B2 values

model.intercept_
#this is the B0 value, intercept

#model here is
# yield = -67.8843597036845 + (0.90608862 x Time) + (-0.06418911 x Temperature)

#performance measures
rsq = model.score(X,y)
rsq

pred = model.predict(X)

mse = mean_squared_error(y,pred)
mse

import math
rmse = math.sqrt(mse)
rmse

from sklearn.model_selection import cross_val_score

score = cross_val_score(model, X, y, scoring='neg_mean_squared_error',cv = 4)
score

"""if R^2 value is greater than normal R^2 value use the model predicted with cross validation to obtain the result

use correct multiple, here cv=4 as here 16 data points 
"""

#Error analysis
'''model.summary()'''

'''import statsmodels.api as sm '''

'''
model = sm.OLS(y,X)
fitted_model = model.fit()
fitted_model.summary()
'''

#Residual Analysis
# 1.pp plot - if all points lie in x=y then the model is good and follows approximation of normal distribution

"""### Logistic Regression

Predict non - payment of draft by customers
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df1 = pd.read_csv("Logistic_Reg.csv")

df2 = pd.read_csv("bank-data.csv")

df1.describe()

sns.pairplot(df1, hue='Outcome')
plt.show()

X = df1.iloc[:,0:3]
y = df1.Outcome

model = LogisticRegression(C = 1e08) # 1e08 = 10^8
model.fit(X,y)

model.intercept_

model.coef_

#model is:
'''
 outcome  = 1 / 1+e**(-z)
  z = -35.50615344 + (2.7957264*Ind_Exp_Act_Score) + (2.75315703*Tran_Speed_Score)	+ (3.51531432*Peer_Comb_Score)
'''

accuracy = model.score(X,y)
accuracy

pred = model.predict(X)

pred_prob = model.predict_proba(X)
pred_prob

predclass = model.predict(X)
# predclass

pd.crosstab(y, predclass)

confusion_matrix(y,predclass)

report = classification_report(y, predclass)
print(report)

"""### Naive Bayes

bayes theorem  P(y/x1..xn) =

Naive Bayes classifier  
- bernoulli - only 2 classes
- Gaussian NB - data is normally distributed
- multinomial - from discrete counts
"""

from sklearn.naive_bayes import GaussianNB

X_train = pd.read_csv("Iris_data.csv")
X_test = pd.read_csv("Iris_test.csv")

X_train.head()

X_train.describe()

X = X_train.iloc[:,0:5]
y = X_train.Species

# model = GaussianNB(X,y)
# model = model.fit(X,y)

from sklearn.model_selection import train_test_split

"""gridSearchCV"""

