import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


start = "2010-01-01"
end = "2019-12-31"

st.header('Stock Market Predictor')
stock =st.text_input('Enter Stock Symnbol', 'AAPL')

yf.pdr_override() 
df = pdr.get_data_yahoo(stock, start=start, end=end)


st.subheader('Stock Data from 2010 - 2019')
st.write(df)

st.subheader('Price vs MA100 vs MA200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label = 'ma100')
plt.plot(ma200, 'b', label = 'ma200')
plt.plot(df.Close, 'g', label = 'Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)


# Splitting Data into Training and Testing 

d = int(len(df)*0.75)
data_training = pd.DataFrame(df['Close'][0:d])
data_testing = pd.DataFrame(df['Close'][d:])


# Load my model
model = load_model('artifact/stock_model.h5')


# get testing data
pas_100_days = data_training.tail(100)
data_test = pd.concat([pas_100_days, data_testing], ignore_index=True)

scaler = MinMaxScaler()
data_test_scale = scaler.fit_transform(data_test)

x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = model.predict(x_test)

scale =1/scaler.scale_
y_predict = y_predict*scale
y_test = y_test*scale

st.subheader('')
st.subheader('')
st.subheader('')
st.subheader('')
st.subheader('Predicted Price vs Original Price')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y_test, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)