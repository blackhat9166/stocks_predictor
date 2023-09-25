import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yahoo
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-06-24'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker: ','AAPL')

df = yahoo.download(user_input, start , end, progress=False)

st.subheader('Data from 2010 - 2023')
st.write(df.describe())


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart Moving Average 100 Days')
moving_avg100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(moving_avg100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart Moving Average 100 & 200 Days')
moving_avg100 = df.Close.rolling(100).mean()
moving_avg200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(moving_avg100, 'r')
plt.plot(moving_avg200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
past_100_days = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df = past_100_days
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Original Vs Predicted Stock Closing Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original_Price')
plt.plot(y_predicted, 'r', label = 'Predicted_Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

