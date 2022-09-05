'''
-Preprocesses stocks and relation data

1. Constructs features for stocks daily data by utilizing fractional differencing to use as model input
    - clears target directory first
    - clean data
        : drop nans, 0's and rows with repeated dates
    - following procedures are conducted on the whole period
    - get cumulative sum of OHLC data per stock
    - find minimum d (differencing quantity) s.t. 
        corresponding Fixed-Width Window FracDiff (FFD) yields 
        p-value below 5% on Augmented-DickeyFuller (ADF) test per stock (also per OHLC - should be of equal length)
    - merge generated stock feature data back with date column
=> saves data to /data/stocks/processed/NASDAQ/ and /data/stocks/processed/NYSE/
=> saves plot of d, ADF Test statistic value, and correlation per stock to /results/ADF_plots/NASDAQ/ and /results/ADF_plots/NYSE/

2. Selection of stock tickers according to following conditions (sequentially)
    - Preconditioned on stock tickers that have succesfully generated FFD (no nans)
    - All conditions below are for date range given as argument
    - remove stocks with nan values
    - remove stocks that have zero volume (were not traded at some day)
    - remove stocks data with repeated date values
    - use AAPL as criterion to select stocks that have same recorded dates data (same rows)
    - (remove stocks traded under $5 at least once - optional, currently commented out)
    - use minimum average close*volume given as argument to select stocks above that criterion
=> Saves selected ticker file to /data/stocks/processed/NASDAQ_tickers.txt, /data/stocks/processed/NYSE_tickers.txt
    
3. Makes industry and wiki relation data
    - updates industry ticker data by removing tickers not in selected ticker file
    - makes industry relation tensor data using selected ticker file and updated industry ticker data
        => saves to /data/relation/industry/NASDAQ_industry_relation.npy and /data/relation/industry/NYSE_industry_relation.npy
    - updates wiki_q file by removing rows with tickers not selected
        => saves to /data/relation/wiki/nasd_wiki_q_selected.csv and /data/relation/wiki/nyse_wiki_q_selected.csv
    - updates wiki connection by removing connections between tickers not in selected ticker file
    - makes wiki relation tensor data using selected ticker file and updated connection
        => saves to /data/relation/wiki/NASDAQ_wiki_relation.npy and /data/relation/wiki/NYSE_wiki_relation.npy
'''

import pandas as pd
import numpy as np
import os
import argparse
import datetime as dt
import json
import copy
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as mpl


def get_FFD_weights(d, thres, data_size):
    weights = [1.0]
    for k in range(1, data_size):
        w = -weights[-1] / k * (d - k + 1)
        if abs(w) < thres:
            break
        
        weights.append(w)
    weights = np.array(weights[::-1]).reshape(-1, 1)
    
    return weights

def FFD(series, d, thres):
    '''
    Fixed Window Fractional Differencing
    '''
    w = get_FFD_weights(d, thres, len(series))
    
    FFD_series = pd.Series()
    for j in range(len(w)-1, len(series)):
        start, end = series.index[j - len(w) + 1], series.index[j]
        FFD_series.loc[end] = np.dot(w.T, series.loc[start:end])[0]
        
    return FFD_series

def construct_frac_diff_features(market_name, thres=1e-5, num_ds=10, use_log_prc=True):
    '''
    thres: weight drop threshold -> decides weights to drop per each timestep data
    num_ds: number of d's between 0 and 1 ( (0, 1], equal spaced) to search for 
    '''
    
    if market_name not in ['NASDAQ', 'NYSE']:
        print('INVALID MARKET NAME')
        quit()
    
    # => Delete all files in target directory first
    data_targ_path = './data/stocks/processed/' + market_name
    plot_targ_path = './results/ADF_plots/' + market_name + '/'
    
    for f in os.listdir(data_targ_path):
        os.remove(data_targ_path + '/' + f)
        
    for f in os.listdir(plot_targ_path):
        os.remove(plot_targ_path + f)
    
    # PER STOCK
    orig_path = './data/stocks/' + market_name
    
    for stock_file in sorted(os.listdir(orig_path)):
        
        print(market_name, stock_file, flush=True)
        cols = ['Open', 'High', 'Low', 'Close']
        stock = pd.read_csv(orig_path + '/' + stock_file)[['Date'] + cols].replace(to_replace=0, value=None).dropna()

        # => Erase rows with duplicate dates
        stock = stock[~stock.index.duplicated(keep='first')]
        
        # => Get cumulative sum of time series
        ohlc_cum = stock[cols].cumsum()
        
        try:
            # => Find minimum d value
            out_features = []
            for i in range(4): # for OHLC
                adf_res = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'n0bs', '95 conf', 'corr'])
                out_feat_cands = []
                
                for d in np.linspace(0, 1, num_ds+1): # for d's
                    if d == 0:
                        continue
        
                    df1 = ohlc_cum.iloc[:, i]
                    if use_log_prc:
                        df1 = np.log(df1)
                        
                    df2 = FFD(df1, d, thres=thres)
                    out_feat_cands.append(df2)
                    
                    corr = np.corrcoef(df1.loc[df2.index], df2)[0, 1]
                    
                    adf_df2 = adfuller(df2, maxlag=1, regression='c', autolag=None)
                    adf_res.loc[d] = list(adf_df2[:4]) + [adf_df2[4]['5%'], corr]
                    
                adf_res[['adfStat', 'corr']].plot(secondary_y='adfStat')
                crit_val = adf_res['95 conf'].mean()
                mpl.axhline(crit_val, linewidth=1, color='r', linestyle='dotted')
                mpl.savefig(plot_targ_path + stock_file[:-4] + '_' + cols[i] + '.png')
                
                for l in range(len(out_feat_cands)):
                    if adf_res['adfStat'].tolist()[l] < crit_val:
                        out_features.append(out_feat_cands[l])
                        break
                
            # => Merge OHLC & Date data and save
            ohlc_features = pd.concat(out_features, axis=1)
            to_save = pd.concat([stock['Date'][ohlc_features.index], ohlc_features], axis=1).reset_index(drop=True)
            to_save.columns = ['Date', 'Open', 'High', 'Low', 'Close']
            to_save.to_csv(data_targ_path + '/' + stock_file)
            
        except Exception as e:
            print(e, flush=True)


def select_stock_tickers(start_date, end_date, min_avg_trd_dollars):
    
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
    
    aapl = pd.read_csv('./data/stocks/NASDAQ/AAPL.csv')
    
    aapl['Date'] = pd.to_datetime(aapl['Date'])
    aapl = aapl.loc[(aapl['Date'] >= start_date) & (aapl['Date'] <= end_date)]
    
    assert aapl.isna().sum().sum() == 0, 'AAPL has nan values in given date range'
    assert 0 not in aapl['Volume'].tolist(), 'AAPL has zero volume in given date range'
    assert True not in aapl.duplicated(subset=['Date']).tolist(), 'AAPL has repeated date rows in given date range'
    
    nasdaq = sorted([f for f in os.listdir('./data/stocks/NASDAQ')])
    nyse = sorted([f for f in os.listdir('./data/stocks/NYSE')])
    
    nasd_ticks = []
    for f in nasdaq:
        print(f)
        
        if f not in os.listdir('./data/stocks/processed/NASDAQ/'):
            continue
        
        processed = pd.read_csv('./data/stocks/processed/NASDAQ/' + f)
        
        processed['Date'] = pd.to_datetime(processed['Date'])
        processed = processed.loc[(processed['Date'] >= start_date) & (processed['Date'] <= end_date)]
        
        if processed.isna().sum().sum() > 0:
            continue
        
        if True in processed.duplicated(subset=['Date']).tolist() or (processed['Date'].tolist() == aapl['Date'].tolist()) == False:
            continue
        
        try:
            df = pd.read_csv('./data/stocks/NASDAQ/' + f)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if df.isna().sum().sum() > 0:
                continue

            if 0 in df['Volume'].tolist():
                continue

            if True in df.duplicated(subset=['Date']).tolist() or (df['Date'].tolist() == aapl['Date'].tolist()) == False:
                continue

            # if len(df[df['Low'] <= 5]) != 0:
            #     continue

            if min_avg_trd_dollars > (df['Close']*df['Volume']).mean():
                continue
            
            nasd_ticks.append(f[:-4])
                
        except Exception as e:
            print(e)
            
    nyse_ticks = []
    for f in nyse:
        print(f)
        
        if f not in os.listdir('./data/stocks/processed/NYSE/'):
            continue
        
        processed = pd.read_csv('./data/stocks/processed/NYSE/' + f)
        
        processed['Date'] = pd.to_datetime(processed['Date'])
        processed = processed.loc[(processed['Date'] >= start_date) & (processed['Date'] <= end_date)]
        
        if processed.isna().sum().sum() > 0:
            continue
        
        if True in processed.duplicated(subset=['Date']).tolist() or (processed['Date'].tolist() == aapl['Date'].tolist()) == False:
            continue
        
        try:
            df = pd.read_csv('./data/stocks/NYSE/' + f)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if df.isna().sum().sum() > 0:
                continue

            if 0 in df['Volume'].tolist():
                continue

            if True in df.duplicated(subset=['Date']).tolist() or (df['Date'].tolist() == aapl['Date'].tolist()) == False:
                continue

            # if len(df[df['Low'] <= 5]) != 0:
            #     continue

            if min_avg_trd_dollars > (df['Close']*df['Volume']).mean():
                continue
            
            nyse_ticks.append(f[:-4])
                
        except Exception as e:
            print(e)
            
    print('Selected', len(nasd_ticks), 'tickers out of', len(nasdaq))
    print('Selected', len(nyse_ticks), 'tickers out of', len(nyse))
    pd.DataFrame(nasd_ticks).to_csv('./data/stocks/processed/NASDAQ_tickers.txt', index=None, header=None)
    pd.DataFrame(nyse_ticks).to_csv('./data/stocks/processed/NYSE_tickers.txt', index=None, header=None)
        

def build_wiki_relation(market_name):
    if market_name == 'NASDAQ':
        market = 'nasd'
    elif market_name == 'NYSE':
        market = 'nyse'
    else:
        print('INVALID MARKET NAME')
        quit()
    
    tic_list = pd.read_csv('./data/relation/wiki/' + market + '_wiki_q.csv')
    
    # => Filter tickers according to selected tickers 
    sel_ticks = list(pd.read_csv('./data/stocks/processed/' + market_name + '_tickers.txt', header=None)[0])
    
    sel_ticks_list = []
    for t in sel_ticks:
        if t in list(tic_list['ticker']):
            sel_ticks_list.append([t, list(tic_list['code'])[list(tic_list['ticker']).index(t)]])
            
    sel_ticks_df = pd.DataFrame(sel_ticks_list, columns=['ticker', 'code'])
    sel_ticks_df.to_csv('./data/relation/wiki/' + market + '_wiki_q_selected.csv')
    
    # => Update connections
    with open('./data/relation/wiki/' + market_name + '_connections.json', 'r') as f:
        old_connections = json.load(f)
        
    connections = copy.deepcopy(old_connections)

    sel_qs = list(sel_ticks_df['code'])
    for sou_item, conns in old_connections.items():
        if sou_item not in sel_qs:
            del connections[sou_item]
            continue
        for tar_item, _ in conns.items():
            if tar_item not in sel_qs:
                del connections[sou_item][tar_item]
                
    # => Make Wiki Relation Tensor
    tic_wiki_file = './data/relation/wiki/' + market + '_wiki_q_selected.csv'
    sel_path_file = './data/relation/wiki/selected_wiki_connections.csv'
                
    # readin tickers
    tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                            skip_header=True)
    print('#tickers selected:', tickers.shape)
    wikiid_ticind_dic = {}
    for ind, tw in enumerate(tickers):
        if not tw[-1] == 'unknown':
            wikiid_ticind_dic[tw[-1]] = ind
    print('#tickers aligned:', len(wikiid_ticind_dic))

    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                              skip_header=False)
    print('#paths selected:', len(sel_paths))
    sel_paths = set(sel_paths[:, 0])

    # readin connections
    print('#connection items:', len(connections))

    # get occured paths
    occur_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    occur_paths.add(path_key)

    # generate
    valid_path_index = {}
    for ind, path in enumerate(occur_paths):
        valid_path_index[path] = ind
    print('#valid paths:', len(valid_path_index))
    for path, ind in valid_path_index.items():
        print(path, ind)

    wiki_relation_embedding = np.zeros(
        [tickers.shape[0], tickers.shape[0], len(valid_path_index) + 1],
        dtype=int
    )

    conn_count = 0
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in valid_path_index.keys():
                    aaa = wikiid_ticind_dic[sou_item]
                    bbb = wikiid_ticind_dic[tar_item]
                    ccc = valid_path_index[path_key]
                    wiki_relation_embedding[wikiid_ticind_dic[sou_item]][wikiid_ticind_dic[tar_item]][valid_path_index[path_key]] = 1
                    conn_count += 1
    print('connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * tickers.shape[0]))

    # handle self relation
    for i in range(tickers.shape[0]):
        wiki_relation_embedding[i][i][-1] = 1
    print(wiki_relation_embedding.shape)
    np.save('./data/relation/wiki/' + market_name + '_wiki_relation.npy', wiki_relation_embedding)

def build_industry_relation(market_name):
    if market_name not in ['NASDAQ', 'NYSE']:
        print('INVALID MARKET NAME')
        quit()
        
    selected_tickers = pd.read_csv('./data/stocks/processed/' + market_name + '_tickers.txt', header=None)[0]
    
    # => Update industry ticker data
    with open('./data/relation/industry/' + market_name + '_industry_ticker.json', 'r') as f:
        old_industry_tickers = json.load(f)
        
    industry_tickers = copy.deepcopy(old_industry_tickers)
        
    for industry, tickers in old_industry_tickers.items():
        for ticker in tickers:
            if ticker not in list(selected_tickers):
                industry_tickers[industry].remove(ticker)
        if len(industry_tickers[industry]) == 0:
            del industry_tickers[industry]
            
    # => Make Industry Relation Tensor
    print('#tickers selected:', len(selected_tickers))
    ticker_index = {}
    for index, ticker in enumerate(selected_tickers):
        ticker_index[ticker] = index
    
    print('#industries: ', len(industry_tickers))
    valid_industry_count = 0
    valid_industry_index = {}
    for industry in industry_tickers.keys():
        if len(industry_tickers[industry]) > 1:
            valid_industry_index[industry] = valid_industry_count
            valid_industry_count += 1
    one_hot_industry_embedding = np.identity(valid_industry_count + 1,
                                                dtype=int)
    ticker_relation_embedding = np.zeros(
        [len(selected_tickers), len(selected_tickers),
            valid_industry_count + 1], dtype=int)
    print(ticker_relation_embedding[0][0].shape)
    for industry in valid_industry_index.keys():
        cur_ind_tickers = industry_tickers[industry]
        if len(cur_ind_tickers) <= 1:
            continue
        ind_ind = valid_industry_index[industry]
        for i in range(len(cur_ind_tickers)):
            left_tic_ind = ticker_index[cur_ind_tickers[i]]
            ticker_relation_embedding[left_tic_ind][left_tic_ind] = \
                copy.copy(one_hot_industry_embedding[ind_ind])
            ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1
            for j in range(i + 1, len(cur_ind_tickers)):
                right_tic_ind = ticker_index[cur_ind_tickers[j]]
                ticker_relation_embedding[left_tic_ind][right_tic_ind] = \
                    copy.copy(one_hot_industry_embedding[ind_ind])
                ticker_relation_embedding[right_tic_ind][left_tic_ind] = \
                    copy.copy(one_hot_industry_embedding[ind_ind])

    # handle industry w/ only one ticker and n/a tickers
    for i in range(len(selected_tickers)):
        ticker_relation_embedding[i][i][-1] = 1
    print(ticker_relation_embedding.shape)
    np.save('./data/relation/industry/' + market_name + '_industry_relation.npy', ticker_relation_embedding)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-s_date', type=str, default='2002-01-01', help='start date; YYYY-MM-DD')
    parser.add_argument('-e_date', type=str, default='2021-12-31', help='end date; YYYY-MM-DD')
    parser.add_argument('-m_trd', type=int, default=5000000, help='minimum average daily dollar volume (close * volume) in period')
    parser.add_argument('-thres', type=float, default=1e-5, help='threshold for dropping FFD weights')
    parser.add_argument('-num_d', type=int, default=10, help='number of evenly spaced d values to search between (0, 1]')
    parser.add_argument('-use_log', type=bool, default=True, help='whether to use log price or not')
    args = parser.parse_args()
    
    construct_frac_diff_features('NASDAQ', thres=args.thres, num_ds=args.num_d, use_log_prc=args.use_log)
    construct_frac_diff_features('NYSE', thres=args.thres, num_ds=args.num_d, use_log_prc=args.use_log)
    
    select_stock_tickers(args.s_date, args.e_date, args.m_trd)
    
    build_wiki_relation('NASDAQ')
    build_wiki_relation('NYSE')
    
    build_industry_relation('NASDAQ')
    build_industry_relation('NYSE')
    
