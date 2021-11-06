# only glucose data is used

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

glucose_csv = '/Users/jonathanlifferth/PycharmProjects/longevity_solutions_ml/glucose.csv'

df = pd.read_csv(glucose_csv)
df = df.drop(columns=['Unnamed: 0'])

# df.plot(figsize=(12, 8))
# plt.title('blood glucose over 6 days')
# plt.ylabel('blood glucose (mg/dl)')
# sns.set_style('whitegrid')
# plt.xticks(rotation=60)
# plt.show()

# create time windows
window_interval = 30  # time in minutes, smallest possible interval is 5 minutes

frame_1 = 'glucose_minus_' + str(window_interval)
frame_2 = 'glucose_minus_' + str(window_interval * 2)
frame_3 = 'glucose_minus_' + str(window_interval * 3)

frame_shift_1 = int(window_interval / 5)
frame_shift_2 = int((window_interval * 2) / 5)
frame_shift_3 = int((window_interval * 3) / 5)
print(frame_shift_1, frame_shift_2, frame_shift_3)

df[frame_1] = df['glucose'].shift(+frame_shift_1)
df[frame_2] = df['glucose'].shift(+frame_shift_2)
df[frame_3] = df['glucose'].shift(+frame_shift_3)

# drop na values
df = df.dropna()
print(df)

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_features=3, random_state=1)

import numpy as np
x1,x2,x3,y=df[frame_1],df[frame_2],df[frame_3],df['glucose']
x1,x2,x3,y=np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
print(final_x)
print(final_x.shape)

# split 70/30 into train and test sets
X_train_size = int(len(final_x) * 0.7)
set_index = len(final_x) - X_train_size
print(set_index)
X_train,X_test,y_train,y_test=final_x[:-set_index],final_x[-set_index:],y[:-set_index],y[-set_index:]

model.fit(X_train, y_train) # random forest
lin_model.fit(X_train, y_train) # linear regression

# Random Forest Regressor
pred = model.predict(X_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8)
plt.plot(pred, label='Random_Forest_Predictions')
plt.plot(y_test, label='Actual Glucose')
plt.legend(loc="upper left")
plt.show()

# Linear Regression
lin_pred = lin_model.predict(X_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 6)
plt.plot(lin_pred, label='Linear_Regression_Predictions')
plt.plot(y_test, label='Actual Glucose')
plt.legend(loc="upper left")
plt.show()


# create weekly report for diabetes patient 0
total_time = len(y_test)
time_out_of_range = (y_test > 200).sum()
percent_in_range = ((total_time - time_out_of_range) / total_time) * 100
pred_out_of_range = (lin_pred > 200).sum()
pred_accuracy = (pred_out_of_range / time_out_of_range) * 100
glucose_max = y_test.max()
glucose_min = y_test.min()
glucose_mean = y_test.mean()


# print(total_time)
# print(time_out_of_range)
print('This week, you spent ', percent_in_range, '% of your time in range')
print('Great job!\n')
print('Your average glucose level this week was : ', glucose_mean)
print('Your maximum value was: ', glucose_max)
print('Your minimum value was: ', glucose_min, '\n')
print('Nudge accurately predicted ', pred_accuracy, '% of your time out of range')

# create mean aggregate model
print(lin_pred.shape)
pred = pred.reshape(-1, 1)
print(pred.shape)
print(y_test.shape)

aggregate_pred = np.mean(np.array([lin_pred, pred]), axis=0)
# print(aggregate_pred)

# Aggregate Regression
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 6)
plt.plot(aggregate_pred, label='Aggregate_Predictions')
plt.plot(y_test, label='Actual Glucose')
plt.legend(loc="upper left")
plt.show()

# evaluate RMSE for each prediction type
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_rf = sqrt(mean_squared_error(pred, y_test))
rmse_lr = sqrt(mean_squared_error(lin_pred, y_test))
rmse_agg = sqrt(mean_squared_error(aggregate_pred, y_test))
print('Mean Squared Error for Random Forest Model is:', rmse_rf)
print('Mean Squared Error for Linear Regression Model is:', rmse_lr)
print('Mean Squared Error for Aggregate Model is:', rmse_agg)

