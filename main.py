import configs
import networks
import trainer

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
import pandas as pd
import os
import json
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

def train(market, relation_type, relation_tensor, raw_data, feature_data, num_stocks):
    suffix = market + '_' + relation_type
    json_to_dir = './models/configs/' + configs.all_configs['airl_hyperparams']['tensorboard_path'].split('/')[-2]
    
    configs_to_save = copy.deepcopy(configs.all_configs)
    
    configs_to_save['airl_hyperparams']['model_save_path']['Actor'] += '_' + suffix + '.pt'
    configs_to_save['airl_hyperparams']['model_save_path']['Discriminator'] += '_' + suffix + '.pt'
    configs_to_save['airl_hyperparams']['tensorboard_path'] += 'train/' + suffix
    configs_to_save['airl_hyperparams']['DEVICE'] = DEVICE
    
    airl_hp = configs_to_save['airl_hyperparams']
    net_hp = configs_to_save['network_hyperparams']
    
    actor_net = networks.Actor(num_stocks, airl_hp['num_sel_stocks'], feature_data.size(dim=-1), relation_tensor.size(dim=-1), DEVICE, net_hp['LSTM_hiddim'], net_hp['LSTM_attention'])
    disc_net = networks.Discriminator(DEVICE, num_stocks, net_hp['Disc_gamma'], **net_hp['g_h_args'])
    
    # TRAINING
    airl_trainer = trainer.AIRL(market, relation_type, raw_data, feature_data, relation_tensor, actor_net, disc_net, **configs_to_save['airl_hyperparams'])
    airl_trainer.train()
    
    # SAVE MODEL CONFIGURATION AS JSON FILE
    configs_to_save['train_start'] = configs_to_save['train_start'].strftime('%Y%m%d')
    configs_to_save['train_end'] = configs_to_save['train_end'].strftime('%Y%m%d')
    configs_to_save['test_start'] = configs_to_save['test_start'].strftime('%Y%m%d')
    configs_to_save['test_end'] = configs_to_save['test_end'].strftime('%Y%m%d')
    
    with open(json_to_dir + '_' + suffix + '.json', 'w') as outfile:
        json.dump(configs_to_save, outfile)


industry_rel_dir = './data/relation/industry/'
wiki_rel_dir = './data/relation/wiki/'
data_dir = './data/stocks/'

nasdaq_industry_relation = torch.tensor(np.load(industry_rel_dir + 'NASDAQ_industry_relation.npy'), dtype=torch.float)
nyse_industry_relation = torch.tensor(np.load(industry_rel_dir + 'NYSE_industry_relation.npy'), dtype=torch.float)
nasdaq_wiki_relation = torch.tensor(np.load(wiki_rel_dir + 'NASDAQ_wiki_relation.npy'), dtype=torch.float)
nyse_wiki_relation = torch.tensor(np.load(wiki_rel_dir + 'NYSE_wiki_relation.npy'), dtype=torch.float)

print('nasdaq industry', nasdaq_industry_relation.size())
print('nyse industry', nyse_industry_relation.size())
print('nasdaq wiki', nasdaq_wiki_relation.size())
print('nyse wiki', nyse_wiki_relation.size())

# Get selected tickers
nasdaq_tickers = list(pd.read_csv(data_dir + 'processed/NASDAQ_tickers.txt', header=None)[0])
nyse_tickers = list(pd.read_csv(data_dir + 'processed/NYSE_tickers.txt', header=None)[0])

# Get raw data & preprocessed feature data
nasdaq_raw_data, nasdaq_feature_data = [], []
nyse_raw_data, nyse_feature_data = [], []
past_days = configs.all_configs['network_hyperparams']['data_seq_len']

start_day = configs.all_configs[configs.mode + '_start']
end_day = configs.all_configs[configs.mode + '_end'] 

# For test, start_day -= data_seq_len
if configs.mode == 'test':
    temp = pd.read_csv(data_dir + 'NASDAQ/' + nasdaq_tickers[0] + '.csv')
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp2 = temp[temp['Date'] >= start_day]
    start_idx = temp2.index.tolist()[0] - past_days
    trade_start_idx = temp2.index.tolist()[0] + 1
    start_day = temp['Date'][start_idx]
    trade_start_day = temp['Date'][trade_start_idx]
    
    test_days = temp[(temp['Date'] >= trade_start_day) & (temp['Date'] <= end_day)]['Date'].reset_index(drop=True).tolist()
    
# Collect NASDAQ raw, feature data
for nasdaq_ticker in nasdaq_tickers:
    print('NASDAQ', nasdaq_ticker)
    raw = pd.read_csv(data_dir + 'NASDAQ/' + nasdaq_ticker + '.csv')
    feature = pd.read_csv(data_dir + 'processed/NASDAQ/' + nasdaq_ticker + '.csv').drop(columns=['Unnamed: 0'])
    
    raw['Date'] = pd.to_datetime(raw['Date'])
    feature['Date'] = pd.to_datetime(feature['Date'])
    raw = raw[(raw['Date'] >= start_day) & (raw['Date'] <= end_day)]['Close'].reset_index(drop=True).pct_change()
    feature = feature[(feature['Date'] >= start_day) & (feature['Date'] <= end_day)].reset_index(drop=True).drop(columns=['Date'])

    raw = np.array(raw)
    nasdaq_raw_data.append(torch.tensor(raw[past_days+1:], dtype=torch.float))
    
    feature = np.array(feature)
    
    feature_train = []
    for d in range(past_days, feature.shape[0]):
        feature_train.append(feature[d - past_days: d, :])
    nasdaq_feature_data.append(torch.tensor(np.stack(feature_train, axis=0), dtype=torch.float))
    
nasdaq_raw_data = torch.stack(nasdaq_raw_data, dim=1)
nasdaq_feature_data = torch.stack(nasdaq_feature_data, dim=1)

# Collect NYSE raw, feature data
for nyse_ticker in nyse_tickers:
    print('NYSE', nyse_ticker)
    raw = pd.read_csv(data_dir + 'NYSE/' + nyse_ticker + '.csv')
    feature = pd.read_csv(data_dir + 'processed/NYSE/' + nyse_ticker + '.csv').drop(columns=['Unnamed: 0'])
    
    raw['Date'] = pd.to_datetime(raw['Date'])
    feature['Date'] = pd.to_datetime(feature['Date'])
    raw = raw[(raw['Date'] >= start_day) & (raw['Date'] <= end_day)]['Close'].reset_index(drop=True).pct_change()
    feature = feature[(feature['Date'] >= start_day) & (feature['Date'] <= end_day)].reset_index(drop=True).drop(columns=['Date'])
    
    raw = np.array(raw)
    nyse_raw_data.append(torch.tensor(raw[past_days+1:], dtype=torch.float))
    
    feature = np.array(feature)
    
    feature_train = []
    for d in range(past_days, feature.shape[0]):
        feature_train.append(feature[d - past_days: d, :])
    nyse_feature_data.append(torch.tensor(np.stack(feature_train, axis=0), dtype=torch.float))
    
nyse_raw_data = torch.stack(nyse_raw_data, dim=1)
nyse_feature_data = torch.stack(nyse_feature_data, dim=1)

print(nasdaq_raw_data.size())
print(nasdaq_feature_data.size())
print(nyse_raw_data.size())
print(nyse_feature_data.size())

if configs.mode == 'train':
    
    # Train NASDAQ INDUSTRY
    
    train('NASDAQ', 'industry', nasdaq_industry_relation, nasdaq_raw_data, nasdaq_feature_data, len(nasdaq_tickers))
    
    # Train NYSE INDUSTRY
    
    train('NYSE', 'industry', nyse_industry_relation, nyse_raw_data, nyse_feature_data, len(nyse_tickers))
    
    # Train NASDAQ WIKI
    
    train('NASDAQ', 'wiki', nasdaq_wiki_relation, nasdaq_raw_data, nasdaq_feature_data, len(nasdaq_tickers))
    
    # Train NYSE WIKI
    
    train('NYSE', 'wiki', nyse_wiki_relation, nyse_raw_data, nyse_feature_data, len(nyse_tickers))


# Generate daily returns of actor & max-sharpe portfolio on test set
def test(market, relation_type, relation_tensor, raw_data, feature_data):
    airl_hp = configs.all_configs['airl_hyperparams']
    model_prefix = airl_hp['model_save_path']['Actor']
    suffix = market + '_' + relation_type
    model_path = model_prefix + '_' + suffix + '.pt'
    
    net_hp = configs.all_configs['network_hyperparams']
    
    rebal = airl_hp['rebal_period']
    num_sel_stocks = airl_hp['num_sel_stocks']
    trans_cost = airl_hp['trans_cost']
    
    raw_data = raw_data.to(DEVICE)
    feature_data = feature_data.to(DEVICE)
    relation_tensor = relation_tensor.to(DEVICE)
    
    actor_cov_var = torch.full(size=(num_sel_stocks,), fill_value=0.5).to(DEVICE)
    actor_cov_mat = torch.diag(actor_cov_var)
    
    actor_net = networks.Actor(raw_data.size(dim=-1), airl_hp['num_sel_stocks'], feature_data.size(dim=-1), relation_tensor.size(dim=-1), DEVICE, net_hp['LSTM_hiddim'], net_hp['LSTM_attention'])
    actor_net.load_state_dict(torch.load(model_path))
    
    actor_net.eval()
    with torch.no_grad():
        sel_feature_data_list = []
        # Select network input data from feature data (dates within trade period are not used) #
        for i in range(0, raw_data.size(dim=0), rebal):
            sel_feature_data_list.append(feature_data[i, ...])
        sel_feature_data = torch.stack(sel_feature_data_list, dim=0)
        
        # Calculate Actor mean_vectors and actor_scores BATCH-wise #
        mean_vectors, actor_scores = actor_net(sel_feature_data, relation_tensor)
        
        expert_returns_list, actor_returns_list = [], []
        # Per each rebalancing date
        for i in range(0, raw_data.size(dim=0), rebal):
            print(test_days[i])
            
            # Exception Case: when num_test_days // rebal != 0, last trade period will be remaining days
            if i + rebal > raw_data.size(dim=0):
                rebal = raw_data.size(dim=0) - i
            
            day_returns = raw_data[i: i + rebal, :] # daily return for all stocks in current trade period
            day_returns[0] -= trans_cost
            sharpes = (252**0.5) * torch.div(torch.mean(day_returns, dim=0), torch.std(day_returns, dim=0)) # sharpe ratios of all stock returns in current trade period
            
            # Calculate Max-Sharpe Weights per each Rebalancing Period #
            _, idx = torch.topk(sharpes, num_sel_stocks)
            sel_returns = torch.index_select(day_returns, 1, idx).detach().cpu().numpy()
            
            exp_rets = expected_returns.mean_historical_return(sel_returns, returns_data=True, compounding=True)
            cov_mat = risk_models.sample_cov(sel_returns, returns_data=True, frequency=rebal)
            ef = EfficientFrontier(exp_rets, cov_mat)
            expert_ws = torch.tensor(list(ef.max_sharpe(risk_free_rate=0).values()), dtype=torch.float).to(DEVICE)
            
            temp_zeroes = torch.zeros_like(sharpes).to(DEVICE)
            expert_weights = temp_zeroes.scatter(0, torch.sort(idx)[0], expert_ws)
        
            # Generate Daily Returns for Max-Sharpe Portfolio #
            expert_returns = torch.squeeze(torch.matmul(day_returns, expert_weights.view(-1, 1)), dim=1)
            expert_returns_list.append(expert_returns)
            
            # Calculate Actor Weights per each Rebalancing Period #
            dist = MultivariateNormal(loc=mean_vectors[i//airl_hp['rebal_period']], covariance_matrix=actor_cov_mat)
            actor_ws = nn.functional.softmax(dist.sample())
            
            _, idx2 = torch.topk(torch.unsqueeze(actor_scores[i//airl_hp['rebal_period']], dim=0), actor_ws.size(dim=-1))
            temp_zeroes2 = torch.zeros_like(torch.unsqueeze(actor_scores[i//airl_hp['rebal_period']], dim=0)).to(DEVICE)
            actor_weights = temp_zeroes2.scatter(1, torch.sort(idx2)[0], torch.unsqueeze(actor_ws, dim=0))
        
            # Generate Daily Returns for Actor Portfolio #
            actor_returns = torch.squeeze(torch.matmul(day_returns, actor_weights.view(-1, 1)), dim=1)
            actor_returns_list.append(actor_returns)
            
        test_expert_returns = torch.cat(expert_returns_list, dim=0).detach().cpu().numpy()
        test_actor_returns = torch.cat(actor_returns_list, dim=0).detach().cpu().numpy()
        
    pd.DataFrame(test_expert_returns, index=test_days, columns=['returns']).to_csv('./results/backtest/' + model_path.split('/')[-1][:-3] + '_EXPERT.csv')
    pd.DataFrame(test_actor_returns, index=test_days, columns=['returns']).to_csv('./results/backtest/' + model_path.split('/')[-1][:-3] + '_ACTOR.csv')

if configs.mode == 'test':
    
    # Test NASDAQ INDUSTRY
    
    test('NASDAQ', 'industry', nasdaq_industry_relation, nasdaq_raw_data, nasdaq_feature_data)
    
    # Test NYSE INDUSTRY
    
    test('NYSE', 'industry', nyse_industry_relation, nyse_raw_data, nyse_feature_data)
    
    # Test NASDAQ WIKI
    
    test('NASDAQ', 'wiki', nasdaq_wiki_relation, nasdaq_raw_data, nasdaq_feature_data)
    
    # Test NYSE WIKI
    
    test('NYSE', 'wiki', nyse_wiki_relation, nyse_raw_data, nyse_feature_data)