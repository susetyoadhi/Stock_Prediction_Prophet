import streamlit as st
import webbrowser
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"#tanggal mulai
TODAY = date.today().strftime("%Y-%m-%d")#tanggal hari ini

# url = "https://stockanalysis.com/stocks/"

st.title('Stock Prediction App')
st.write('website untuk memprediksi harga saham suatu perusahaan untuk beberapa waktu kedepan')

# stocks = ('GOOG', 'AAPL', 'MSFT', 'ADBE', 'TSLA')
# selected_stock = st.selectbox('Select dataset for prediction', stocks)
selected_stock = st.text_input('masukkan ticker company')#textbox untuk memasukkan ticker company
st.subheader('company ticker list: https://stockanalysis.com/stocks/')#link ticker company

# lama tahun yang ingin diprediksi
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# jika nama company dimasukkan
if selected_stock:
	# mengambil data
	data_load_state = st.text('Loading data...')
	data = load_data(selected_stock)
	data_load_state.text('Loading data... done!')
	st.subheader('Data Harga')
	# menampilkan data harga terakhir
	st.write(data.tail())


	# menampilkan data dalam bentuk grafik
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()

	# train data
	# Predict forecast with Prophet.
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	# modul fbprophet
	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	# Show and plot forecast
	st.subheader('prediction data')
	st.write(forecast.tail())
		
	st.write(f'Forecast plot for {n_years} years')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.write("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)