import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

stocks = ('AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'GOOGL')
sstock = st.selectbox('Select dataset for prediction', stocks)

years = st.slider("Years of Prediction:", 1, 4)
time = years * 365


def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name ='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name ='stock_close'))
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {'Date':'ds', "Close":'y'})

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(times = time)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())
