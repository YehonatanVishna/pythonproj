#version 0.6.2
#Model with 3 Categories for zone1 only
from secrets import randbelow
from xmlrpc.client import DateTime
import numpy as np 
import pandas as pd;
import sklearn.metrics 
from datetime import datetime, timedelta;
from sklearn import datasets 
from matplotlib import pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split as split 
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import confusion_matrix as confusion 
 
def convert_DateTime_to_ints(dates):
    newDates = np.empty(6);
    for date in dates:
        time_passed = (datetime.strptime(date, '%m/%d/%Y %H:%M'))
        newDates = np.vstack((newDates, np.array([time_passed.day, time_passed.month, time_passed.year, time_passed.hour, time_passed.minute, time_passed.weekday()])))
    return newDates[1:-1]

panda_df = pd.read_csv("Tetuan City power consumption.csv");
date_before_proccesing = panda_df["DateTime"].values
times = convert_DateTime_to_ints(date_before_proccesing);
day_of_the_month = times[0:-1,0]
months = times[0:-1,1]
years = times[0:-1,2]
hours = times[0:-1,3]
minuets = times[0:-1,4]
day_of_the_week = times[0:-1,5]
tempreture = panda_df["Temperature"].values;
humidity = panda_df["Humidity"].values;
wind_speed = panda_df["Wind Speed"].values;
general_diffuse_flows = panda_df["general diffuse flows"].values;
diffuse_flows = panda_df["diffuse flows"].values;
zone1 = panda_df["Zone 1 Power Consumption"].values;
zone2 = panda_df["Zone 2  Power Consumption"].values;
zone3 = panda_df["Zone 3  Power Consumption"].values;
x = np.array([day_of_the_month, months, years, hours,  minuets  , tempreture , wind_speed ,  general_diffuse_flows ,  diffuse_flows , day_of_the_week]).T


y = np.array([zone1, zone2, zone3]).T
Avereged_X = np.empty(10)
Avereged_Y = np.empty(3)
sum = x[0]
avregeFactor =6
for v in range(avregeFactor,len(x[0])+avregeFactor,avregeFactor):
    x_avg = np.array([x[0][v-avregeFactor], x[1][v-avregeFactor], x[2][v-avregeFactor], x[3][v-avregeFactor], x[4][v-avregeFactor],np.average(x[5][(v-avregeFactor):v]), np.average(x[6][(v-avregeFactor):v]),np.average(x[7][(v-avregeFactor):v]), np.average(x[8][(v-avregeFactor):v]), x[9][v-avregeFactor]])
    Avereged_X = np.vstack((Avereged_X ,[x_avg]))
    y_avg = np.array([np.average(y[(v-avregeFactor):v].T[0]), np.average(y[(v-avregeFactor):v].T[1]), np.average(y[(v-avregeFactor):v].T[2])])
    Avereged_Y = np.vstack((Avereged_Y ,[y_avg]))
Avereged_X = Avereged_X[1:-1]
Avereged_Y = Avereged_Y[1:-1]

print(len(Avereged_X))
print(len(Avereged_Y))
y_zone1 = np.array(Avereged_Y[::,0])
print(Avereged_Y[0:-1].__contains__(Avereged_Y[0]))
print(Avereged_Y[0:-1].__contains__(Avereged_Y[-1]))
# plt.scatter(Avereged_X[0:-1,3], y_zone1)
# plt.show();
low = 25000; medium = 35000;
y1 = y_zone1*0; y1[np.argwhere(y_zone1 <= low)] = 1;
y2 = y_zone1*0; y2[np.argwhere((y_zone1 > low) & (y_zone1 <= medium))] = 1;
y3 = y_zone1*0; y3[np.argwhere(y_zone1 > medium)] = 1;
y_categories = np.array([y1,y2,y3]).T


# plt.scatter(Avereged_X.T[3], Avereged_Y.T[0], color="blue")
# plt.scatter(Avereged_X.T[3], Avereged_Y.T[1], color="red")
# plt.scatter(Avereged_X.T[3], Avereged_Y.T[2], color= "yellow")
# plt.show();
# plt.scatter(Avereged_X.T[1], Avereged_Y.T[0])
# plt.title("power consomption in zone1 as a function of the month")
# plt.show();
# plt.scatter(Avereged_X.T[5], Avereged_Y.T[0])
# plt.title("power consomption in zone1 as a function of the tempreture")
# plt.show();
# plt.scatter(Avereged_X.T[-1], Avereged_Y.T[0])
# plt.title("power consomption in zone1 as a function of the the day of the week")
# plt.show();
print(len(y_categories))
xtrain,xtest, ytrain,ytest = split(Avereged_X,y_categories, train_size=0.75, random_state=1)

model = tf.keras.models.Sequential();
model.add(tf.keras.layers.Dense(units= 6))
model.add(tf.keras.layers.Dense(units= 3))
model.add(tf.keras.layers.Activation(np.round))

model.compile(optimizer='adam',loss='mean_squared_error')
v= model.fit(xtrain,ytrain,epochs=10**0, verbose=1)
res= model.predict(xtest)
print(mse(res.T[0].T,ytest.T[0].T))
plt.plot(range(len(v.history['loss'])), v.history['loss']) 
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()