#version 0.7.2
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
x = np.array([day_of_the_month *0, months, years*0, hours,  minuets *0 , tempreture , wind_speed  ,  general_diffuse_flows *0 ,  diffuse_flows , day_of_the_week]).T


y = np.array([zone1, zone2, zone3]).T
Avereged_X = np.empty(10)
Avereged_Y = np.empty(3)
sum = x[0]
avregeFactor =1
for v in range(avregeFactor,len(x[0])+avregeFactor,avregeFactor):
    x_avg = np.array([x[0][v-avregeFactor], x[1][v-avregeFactor], x[2][v-avregeFactor], x[3][v-avregeFactor], x[4][v-avregeFactor],np.average(x[5][(v-avregeFactor):v]), np.average(x[6][(v-avregeFactor):v]),np.average(x[7][(v-avregeFactor):v]), np.average(x[8][(v-avregeFactor):v]), x[9][v-avregeFactor]])
    Avereged_X = np.vstack((Avereged_X ,[x_avg]))
    y_avg = np.array([np.average(y[(v-avregeFactor):v].T[0]), np.average(y[(v-avregeFactor):v].T[1]), np.average(y[(v-avregeFactor):v].T[2])])
    Avereged_Y = np.vstack((Avereged_Y ,[y_avg]))
Avereged_X = Avereged_X[1:-1]
Avereged_Y = Avereged_Y[1:-1]


y_zone1 = np.array(Avereged_Y[::,0])
low = 25000; medium = 35000;
y1 = y_zone1*0; y1[np.argwhere(y_zone1 <= low)] = 1;
y2 = y_zone1*0; y2[np.argwhere((y_zone1 > low) & (y_zone1 <= medium))] = 1;
y3 = y_zone1*0; y3[np.argwhere(y_zone1 > medium)] = 1;
y_categories = np.array([y1,y2,y3]).T


xtrain,xtest, ytrain,ytest = split(Avereged_X,y_categories, train_size=0.75, random_state=1)

model = tf.keras.models.Sequential();

model.add(tf.keras.layers.Dense(units= 3))
# model.add(tf.keras.layers.Dense(units= 3))
model.add(tf.keras.layers.Activation(tf.nn.elu))

model.add(tf.keras.layers.Dense(units= 3))
model.add(tf.keras.layers.Activation(tf.nn.softmax))





model.compile(optimizer='adam',loss='mean_squared_error')
# model.optimizer.lr = 4 * 10**(-2)
v= model.fit(xtrain,ytrain,epochs=10**1, verbose=1)
res= model.predict(xtest)
print(np.round(res.T[0].T))
print(ytest.T[0].T)
print(mse(np.round(res.T[0].T),ytest.T[0].T))
plt.plot(range(len(v.history['loss'])), v.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
state = np.array([0,1,2]); con_ytest = np.dot(ytest,state)
con_res = np.dot(np.round(res),state)
con1 = confusion(con_ytest,con_ytest)
print(len(con_res)," ",len(con_ytest))
con2 = confusion(con_res,con_ytest)
print(con1)
print()
print(con2)
percision = (con2[0][0] + con2[1][1] + con2[2][2])/len(con_res)
print(percision)
# arr = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]).T
# print(arr)