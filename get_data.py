'''
-Gets daily stock data from yahoo finance
for NASDAQ and NYSE tickers denoted in 
/data/NASDAQ_tickers.txt and /data/NYSE_tickers.txt

-Saves them to /data/stocks/NASDAQ/ and /data/stocks/NYSE/
'''

import yfinance as yf
import pandas as pd

nyse_tickers = pd.read_csv('./data/NYSE_tickers.txt', header=None)[0].tolist()
nasdaq_tickers = pd.read_csv('./data/NASDAQ_tickers.txt', header=None)[0].tolist()

for ticker in nyse_tickers:
    print('NYSE, ticker:', ticker)
    try:
        data = yf.Ticker(ticker)
        data = data.history(period='max')

        data.to_csv('./data/stocks/NYSE/' + ticker + '.csv')

    except Exception as e:
        print(e)

for ticker in nasdaq_tickers:
    print('NASDAQ, ticker:', ticker)
    try:
        data = yf.Ticker(ticker)
        data = data.history(period='max')

        data.to_csv('./data/stocks/NASDAQ/' + ticker + '.csv')

    except Exception as e:
        print(e)