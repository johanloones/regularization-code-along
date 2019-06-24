# --------------
## Load the data
import pandas as pd
sales_data=pd.read_csv(path)
sales_data.head()
## Split the data and preprocess
#Split it into train and test sets using the source column.
train=sales_data[sales_data.source=='train'].copy()
test=sales_data[sales_data.source=='test'].copy()

train.drop(columns='source',inplace=True)
test.drop(columns='source',inplace=True)

## Baseline regression model
#Create a baseline regression model with features Item_Weight', 'Item_MRP', 'Item_Visibility and find out the mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
alg1=LinearRegression(normalize=True)

# Train test split
from sklearn.model_selection import train_test_split

# Features Item_Weight', 'Item_MRP', 'Item_Visibility
X1=train.loc[:,['Item_Weight', 'Item_MRP', 'Item_Visibility']]
x_train1,x_val1,y_train1,y_val1=train_test_split(X1,train.Item_Outlet_Sales,test_size=0.3,random_state=45)

#X_test_Weight_MRP_Visibility=X_test.loc[:,['Item_Weight', 'Item_MRP', 'Item_Visibility']]

# Fit model to data
alg1.fit(x_train1,y_train1)
y_pred1=alg1.predict(x_val1)

# Find mean_squared_error
from sklearn.metrics import mean_squared_error,r2_score
mse1=mean_squared_error(y_val1,y_pred1)
print('mse1',mse1)
print('r2 score1',r2_score(y_val1,y_pred1))

## Effect on R-square if you increase the number of predictors
X2= train.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1)
x_train2,x_val2,y_train2,y_val2=train_test_split(X2,train.Item_Outlet_Sales,test_size=0.3,random_state=45)
# Fit model to data
alg2=LinearRegression(normalize=True)
alg2.fit(x_train2,y_train2)
y_pred2=alg2.predict(x_val2)
# Find mean_squared_error
mse2=mean_squared_error(y_val2,y_pred2)
print('mse2',mse2)
print('r2 score2',r2_score(y_val2,y_pred2))

## Effect of decreasing feature from the previous model
X3= train.drop(['Item_Outlet_Sales','Item_Identifier','Item_Visibility','Outlet_Years'],axis=1)
x_train3,x_val3,y_train3,y_val3=train_test_split(X3,train.Item_Outlet_Sales,test_size=0.3,random_state=45)
# Fit model to data
alg3=LinearRegression(normalize=True)
alg3.fit(x_train3,y_train3)
y_pred3=alg3.predict(x_val3)
# Find mean_squared_error
mse3=mean_squared_error(y_val3,y_pred3)
print('mse3',mse3)
print('r2 score3',r2_score(y_val3,y_pred3))

## Detecting hetroskedacity
import matplotlib.pyplot as plt
plt.scatter(y_pred2,(y_val2-y_pred2))
plt.hlines(y=0,xmin=-1000,xmax=5000)
plt.title('Residual Plot')
plt.xlabel('Predicted Value')
plt.ylabel('Residuals')

## Model coefficients
coef = pd.Series(alg2.coef_,x_train2.columns).sort_values()
plt.figure(figsize=(10,10))
coef.plot(kind='bar',title='Model Coef')

## Ridge regression and Lasso regression
l1=Lasso(alpha=0.01).fit(x_train2,y_train2)
l2=Ridge(alpha=0.05).fit(x_train2,y_train2)

l1_pred2=l1.predict(x_val2)
l2_pred2=l2.predict(x_val2)

# mean sq error
l1_mse2=mean_squared_error(y_val2,y_pred2)
l2_mse2=mean_squared_error(y_val2,y_pred2)

print(' lasso mse',l1_mse2)
print('lasso r2 score3',r2_score(y_val3,l1_pred2))
print(' ridge mse',l2_mse2)
print(' ridge r2 score3',r2_score(y_val3,l2_pred2))


## Cross validation
from sklearn import model_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

y_train=y_train2
X_train=x_train2
y_test=y_val2
X_test=x_val2

alpha_vals_lasso=[0.01,0.05,0.5,5]
alpha_vals_ridge=[0.01,0.05,0.5,5,10,15,25]

#Instantiate lasso and ridge
ridge_model=Ridge()
lasso_model=Lasso()

#Grid search lasso and ridge
ridge_grid=GridSearchCV(estimator=ridge_model,param_grid=dict(alpha=alpha_vals_ridge))
lasso_grid=GridSearchCV(estimator=lasso_model,param_grid=dict(alpha=alpha_vals_lasso))

#Fit and predict lasso_grid, ridge_grid
lasso_grid.fit(X_train,y_train)
ridge_grid.fit(X_train,y_train)

ridge_pred=ridge_grid.predict(X_test)
lasso_pred=lasso_grid.predict(X_test)

#rmse lasso_grid, ridge_grid
ridge_rmse=np.sqrt(mean_squared_error(ridge_pred,y_test))
lasso_rmse=np.sqrt(mean_squared_error(lasso_pred,y_test))

#print better model
best_model,Best_Model=('LASSO',lasso_grid) if lasso_rmse< ridge_rmse else ('RIDGE',ridge_grid)

if best_model=='LASSO':
  print(best_model,Best_Model.best_params_,lasso_rmse,r2_score(y_test,lasso_pred))
else:
  print(best_model,Best_Model.best_params_,ridge_rmse,r2_score(y_test,ridge_pred))
  
coef = pd.Series(Best_Model.best_estimator_.coef_,X_train.columns).sort_values()
plt.figure(figsize=(10,10))
coef.plot(kind='bar',title=best_model + ' Model Coef')




