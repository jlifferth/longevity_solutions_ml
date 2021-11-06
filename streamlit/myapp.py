import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.write("""
## Simple Stock Price App
""")
# define ticker symbol
option = st.selectbox(
    'Enter ticker symbol or select from list',
    ('AAPL', 'GOOGL', 'CHRS', 'CX')
)
st.write('You selected:', option)

# define time frame
start = st.selectbox(
    'Select time frame of interest',
    ('1W', '1M', '6M', '1Y', '5Y')
)
st.write('You selected:', start)
FORMAT = '%Y-%m-%d'
end = datetime.now().strftime(FORMAT)


def select_day(selection):
    if selection == '1W':
        output = 7
    elif selection == '1M':
        output = 31
    elif selection == '6M':
        output = 186
    elif selection == '1Y':
        output = 365
    elif selection == '5Y':
        output = 1825
    else:
        output = 365
    return output


time_range = int(select_day(start))
print(time_range)
end = datetime.strptime(end, '%Y-%m-%d')
time_window = timedelta(days=time_range)
start_time = end - time_window
start_time = start_time.strftime(FORMAT)
st.write('Start time:', start_time)


st.write("""
Shown are the stock **closing price** and **volume** of {}
""".format(option))


# get ticker data
tickerData = yf.Ticker(option)
# obtain historical prices

tickerDf = tickerData.history(period='1d', start=start_time, end=end)

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)
st.write("""
## Volume
""")
st.line_chart(tickerDf.Volume)
