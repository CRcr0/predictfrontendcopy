# pip install streamlit fbprophet yfinance plotly
# streamlit run main.py
# python 3 -m venv virtual
# source virtual/bin/activate

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"  #改合适的日期，分疫情前后数据预测，2020年几月为开始；再出一个没有疫情影响的预测
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Predict Foreign Exchange Rate from 6893 Big Data Team 41')

stocks = ('USDCNY=X', 'USDJPY=X', 'USDEUR=X', 'DOGE-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4) #最好改成predict多少天 改下面一行就行
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

#配合前面直接能得到data。怎么找到和本地算法相同的features 怎么从雅虎取到合适的值。
#st.subheader('Raw data')
st.subheader('Existing data with features for prediction')
st.write(data.tail())


def plot_raw_data():
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
     fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
     st.plotly_chart(fig)


plot_raw_data()

#设计控制每天forecast的频率


# Predict forecast with Prophet.
df_train = data[['Date','Close']]     #挑选需要训练的数据
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})   #找fbprophet网站学习。画出了不同的attribute来分析

#说是用了sklearn模型训练
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)  #预测的future数据集
forecast = m.predict(future)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#修改figure放在论文里体现得出数据的效果

#试试heroku部署
#先git init