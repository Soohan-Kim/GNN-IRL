import numpy as np
import os

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

import networks

import torch
from torch import nn, optim
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal

from torch.utils.tensorboard import SummaryWriter
import random

from scipy import integrate
import math

import time

class AIRL:

    def __init__(self, market, relation_type, raw_data, feature_data, relation_tensor, actor_net, disc_net, **hyperparameters):
        '''
        actor_net, disc_net: all instances (initialized outside this class)
        raw_data: close to close return data for stocks within training period
            -torch tensor of size (num training days - num past days - 1, num stocks)
        feature_data: frac-diff preprocessed data
            -torch tensor of size (num training days - num past days, num stocks, num past days, num_features)
        relation_tensor: torch tensor of size (num stocks, num stocks, num relations)
        '''
        
        np.random.seed(80)
        random.seed(80)
        
        self.market = market
        self.relation_type = relation_type
        
        self._init_hyperparams(hyperparameters)

        self.raw_data = raw_data.to(self.device)
        self.data = feature_data.to(self.device)
        self.A = relation_tensor.to(self.device)

        self.actor_net, self.disc_net = actor_net, disc_net
        
        self.actor_optim = Adam(self.actor_net.parameters(), lr=self.actor_init_lr)
        self.disc_optim = SGD(self.disc_net.parameters(), lr=self.disc_init_lr)

        self.cov_var = torch.full(size=(self.num_sel_stocks,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var)
        
        self.cov_inv = torch.inverse(self.cov_mat).to(self.device)

        self.disc_loss_fn = nn.BCELoss()

        self.lr_scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.disc_optim, mode='min', factor=0.5, threshold_mode='rel', threshold=0.01, verbose=True, patience=5
        )

        self.lr_scheduler_P = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optim, mode='min', factor=0.5, threshold_mode='rel', threshold=0.01, verbose=True, patience=5
        )
        
    def get_starting_points(self):
        self.start_ids = []
        for _ in range(self.batch_size):
            start_id = random.randint(0, self.data.shape[0] - 2*self.rebal_period - 1)
            while start_id in self.start_ids:
                start_id = random.randint(0, self.data.shape[0] - 2*self.rebal_period - 1)
            self.start_ids.append(start_id)
            
    def get_sharpe_ratios(self, next_id=None):
        
        if next_id == None:
            sharpes = []
            for start_id in self.start_ids:
                cur_data = self.raw_data[start_id: start_id + self.rebal_period, :]
                sharpe = (252**0.5) * torch.div(torch.mean(cur_data, dim=0), torch.std(cur_data, dim=0))
                sharpes.append(sharpe)
            
            true_scores = torch.stack(sharpes, dim=0).to(self.device)    
        
            return true_scores    
        
        else:
            cur_data = self.raw_data[next_id: next_id + self.rebal_period, :]
            sharpe = (252**0.5) * torch.div(torch.mean(cur_data, dim=0), torch.std(cur_data, dim=0))
            
            return sharpe
            
    def rank_loss_fn(self, stock_scores):
        '''
        -Calculates pairwise rank loss of predicted stock scores and true sharpe ratios of stocks on trading period
        -Aims to learn the ranking order of stocks per given dates (starting point of trading period)
        -stock_scores: torch tensor of shape (batch size, num stocks)
        '''
        true_scores = self.get_sharpe_ratios()
        
        pairwise_rank_losses = []
        for i in range(stock_scores.size(dim=1)):
            stock_scores_i = torch.unsqueeze(stock_scores[:, i], dim=1).repeat(1, stock_scores.size(dim=1))
            true_scores_i = torch.unsqueeze(true_scores[:, i], dim=1).repeat(1, true_scores.size(dim=1))
            
            pairwise_diff = (stock_scores_i - stock_scores) * (true_scores - true_scores_i)
            zero_tensor = torch.zeros_like(pairwise_diff).to(self.device)
            pairwise_loss_i = torch.max(torch.stack((pairwise_diff, zero_tensor), dim=2), dim=2)[0]
            pairwise_rank_losses.append(pairwise_loss_i)
            
        rank_loss = torch.sum(torch.cat(pairwise_rank_losses, dim=1), dim=1)
        
        return torch.mean(rank_loss)
        
    def train(self):
        '''
        Trains actor (policy) and discriminator networks
        '''

        best_disc_loss, best_actor_loss = 9999999999, 9999999999
        writer = SummaryWriter(self.tensorboard_path)

        start_time = time.time()
        
        # Set starting points for S's in (S, A, S') pairs to be collected 
        ## [set once for expert - in line with AIRL paper algorithm]
        self.get_starting_points()
        
        # Collect Expert Trajectories (except log_prob -> calculated per epoch for updated policy)
        expert_state_pairs, expert_actions = self.get_expert_demo()
        
        for epoch in range(1, self.iterations + 1):
            
            epoch_start_time = time.time()
            
            print(self.market, self.relation_type, 'EPOCH', epoch, '/', self.iterations, flush=True)
            
            # Set starting points for S's in (S, A, S') pairs to be collected [renewed per epoch for policy]
            self.get_starting_points()
                
            # Collect Policy Trajectories
            state_pairs, actions, log_probs = self.rollout()
            
            # Get expert_log_probs (calculated from current policy and expert_actions)
            expert_log_probs_list = []
            for b in range(self.batch_size):
                expert_log_prob = self.calc_log_prob(self.no_extension_expert_actions[b], self.mean_vectors[b])
                expert_log_probs_list.append(expert_log_prob)
            expert_log_probs = torch.stack(expert_log_probs_list, dim=0)
            
            # print(expert_log_probs)
            # print(log_probs)
            
            ## Update Discriminator ##
            
            # Calculate Expert Loss
            labels_e = torch.full(
                size=log_probs.size(),
                fill_value=1.0,
                dtype=torch.float
            ).to(self.device)
            
            self.disc_optim.zero_grad()
            
            expert_outs = self.disc_net(expert_log_probs, expert_state_pairs, expert_actions)
            expert_loss = self.disc_loss_fn(expert_outs, labels_e) 
            
            # Calculate Policy Loss
            labels_p = torch.full(
                size=log_probs.size(),
                fill_value=0.0,
                dtype=torch.float
            ).to(self.device)
            
            policy_outs = self.disc_net(log_probs, state_pairs, actions)
            policy_loss = self.disc_loss_fn(policy_outs, labels_p)
            
            #quit()
            
            disc_loss = expert_loss + policy_loss
            disc_loss.backward(retain_graph=True)
            self.disc_optim.step()
            
            print('Discriminator Expert Loss:', expert_loss.item(), flush=True)
            print('Discriminator Policy Loss:', policy_loss.item(), flush=True)
            print('Discriminator Loss:', disc_loss.item(), flush=True)
            
            self.lr_scheduler_D.step(disc_loss.item())
            
            
            ## Update Policy ##
            
            self.actor_optim.zero_grad()
            
            rewards = self.disc_net.estimate_reward(state_pairs, actions)
            
            actor_loss = -torch.mean(log_probs * rewards)
            # entropy actor loss
            #actor_loss = -torch.mean(torch.log(policy_outs) - torch.log(1 - policy_outs))
            #actor_loss = -torch.mean(rewards - log_probs)
            
            rank_loss = self.rank_loss_fn(state_pairs[:, :-1, 0])
            
            actor_total_loss = actor_loss + self.alpha * rank_loss
            
            print('Actor Loss:', actor_loss.item(), flush=True)
            print('Rank Loss:', rank_loss.item(), flush=True)
            print('Actor Net Total Loss:', actor_total_loss.item(), flush=True)
            
            actor_total_loss.backward()
            self.actor_optim.step()
            
            self.lr_scheduler_P.step(actor_total_loss.item())
            
            print('Current Discriminator LR:', self.disc_optim.param_groups[0]['lr'], flush=True)
            print('Current Actor LR:', self.actor_optim.param_groups[0]['lr'], flush=True)
            
            
            # Save Model
            if disc_loss.item() < best_disc_loss:
                best_disc_loss = disc_loss.item()
                torch.save(self.disc_net.state_dict(), self.model_save_path['Discriminator'])

            if actor_loss.item() < best_actor_loss:
                best_actor_loss = actor_total_loss.item()
                torch.save(self.actor_net.state_dict(), self.model_save_path['Actor'])

            # Log Losses to Tensorboard
            writer.add_scalars('Discriminator_Loss', {
                'Expert Loss': expert_loss.item(), 'Policy Loss': policy_loss.item()
            }, epoch)
            
            writer.add_scalars('Actor_Net_Loss', {
                'Actor Loss': actor_loss.item(), 'Rank Loss': rank_loss.item()
            }, epoch)
            
            writer.add_scalars('Loss', {
                'Discriminator_Loss': disc_loss.item(), 'Actor_Net_Loss': actor_total_loss.item()
            }, epoch)

            # Log LR History to Tensorboard
            writer.add_scalars('LR', {
                'Discriminator_LR': self.disc_optim.param_groups[0]['lr'], 'Actor_LR': self.actor_optim.param_groups[0]['lr']
            }, epoch)
            
            print('Epoch Running time:', time.time() - epoch_start_time, flush=True)
            
        writer.close()

        print('Running Time:', time.time() - start_time)
    
    def rollout(self):
        '''
        Returns policy-generated state_pairs (s, s')'s, actions a's and log_probs log(pi(a|s))'s
            -state_pairs: torch tensor of size (batch size, num_stocks + 1, 2)
            -actions: torch tensor of size (batch size, num_stocks) [non selected stocks are 0-filled]
            -log_probs: torch tensor of size (batch size, 1)
            
        Also updates self.mean_vectors to store current policy mean_vectors per iteration
        '''
        
        actions_list, log_probs_list = [], []
        self.mean_vectors = []
        
        x_list, x_dash_list = [], []
        # For each starting day
        for start_id in self.start_ids:
            # Get s
            x = torch.unsqueeze(self.data[start_id, ...], dim=0)
            x_list.append(x)
            
        batch_x = torch.cat(x_list, dim=0)
            
        mean_vector, s_temp = self.actor_net(batch_x, self.A)
        batch_s = torch.cat((s_temp, torch.zeros(self.batch_size, 1).to(self.device)), dim=1)
        
        portfolio_returns_list = []
        for b in range(self.batch_size):
            self.mean_vectors.append(mean_vector[b])
            
            # Get a and log(pi(a|s))
            action, log_prob = self.sample_action(mean_vector[b], torch.unsqueeze(s_temp[b], dim=0))
            actions_list.append(action)
            log_probs_list.append(log_prob)
            
            # Get s'
            start_id = self.start_ids[b]
            x_dash = torch.unsqueeze(self.data[start_id+1, ...], dim=0)
            x_dash_list.append(x_dash)
            
            period_returns = torch.sum(self.raw_data[start_id: start_id + self.rebal_period, :], dim=0) - self.trans_cost
            portfolio_return = torch.matmul(period_returns, action)
            portfolio_returns_list.append(portfolio_return.view(1, 1))
            
        batch_x_dash = torch.cat(x_dash_list, dim=0)
        _, s_dash_temp = self.actor_net(batch_x_dash, self.A)
        
        portfolio_returns = torch.cat(portfolio_returns_list, dim=0)
        batch_s_dash = torch.cat((s_dash_temp, portfolio_returns), dim=1)
            
        state_pairs = torch.stack((batch_s, batch_s_dash), dim=2)
        actions = torch.stack(actions_list, dim=0)
        log_probs = torch.stack(log_probs_list, dim=0)
        
        return state_pairs, actions, log_probs
    
    def sample_action(self, mean_vector, cur_scores):
        '''
        Returns sampled action and log prob given mean vector
            -mean_vector: torch tensor of size (num_sel_stocks,)
            -cur_scores: torch tensor of size (1, num stocks)
            
            -action: torch tensor of size (num_stocks,)
            -log_prob: torch tensor of size (1,)
        '''
        
        dist = MultivariateNormal(loc=mean_vector, covariance_matrix=self.cov_mat)
        action_temp = dist.sample()
        
        a = nn.functional.softmax(action_temp)
        log_prob = self.calc_log_prob(a, mean_vector)
        
        # print('POLICY ##################')
        # print(a)
        
        ###### action 0-filling for non selected stocks ######
        _, idx = torch.topk(cur_scores, a.size(dim=-1))
        temp_zeroes = torch.zeros_like(cur_scores, requires_grad=True).to(self.device)
        action = temp_zeroes.scatter(1, torch.sort(idx)[0], torch.unsqueeze(a, dim=0))
        
        return torch.squeeze(action, dim=0), log_prob
    
    def get_expert_demo(self):
        '''
        Returns expert state_pairs (s, s')'s, actions a's and log_probs log(pi(a|s))'s
            -state_pairs: torch tensor of size (batch size, num_stocks + 1, 2)
            -actions: torch tensor of size (batch size, num_stocks) [non selected stocks are 0-filled]
            
        => actions are selected by the following steps:
            1. Select the top num_sel_stocks sharpe-ratio stocks for given trading period
            2. Calculate weights for the selected stocks s.t. they maximize sharpe ratio of portfolio for given trading period
            3. 0-fill the weight vector for non selected stocks
        '''
        
        state_pairs_list, actions_list = [], []
        sharpes = self.get_sharpe_ratios()
        self.no_extension_expert_actions = []
        # For each starting day
        for i in range(len(self.start_ids)):        
            start_id = self.start_ids[i]
            returns = self.raw_data[start_id: start_id + self.rebal_period, :]
            
            # Selection
            _, idx = torch.topk(sharpes[i, :], self.num_sel_stocks)
            sel_returns = torch.index_select(returns, 1, idx).detach().cpu().numpy()
            
            # Max-Sharpe Weights
            exp_rets = expected_returns.mean_historical_return(sel_returns, returns_data=True, compounding=True)
            cov_mat = risk_models.sample_cov(sel_returns, returns_data=True, frequency=self.rebal_period)
            
            ef = EfficientFrontier(exp_rets, cov_mat)
            w = ef.max_sharpe(risk_free_rate=0)
            a = torch.tensor(list(w.values()), dtype=torch.float).to(self.device)
            
            self.no_extension_expert_actions.append(a)
            #print(a)
            
            # Extension
            temp_zeroes = torch.zeros_like(sharpes[i, :], requires_grad=True).to(self.device)
            action = temp_zeroes.scatter(0, torch.sort(idx)[0], a)
            actions_list.append(action)
            
            # Get s, s'
            s = torch.cat((sharpes[i, :], torch.zeros(1).to(self.device)), dim=-1)
            next_sharpes = self.get_sharpe_ratios(next_id=start_id + self.rebal_period)
            period_returns = torch.sum(self.raw_data[start_id: start_id + self.rebal_period, :], dim=0) - self.trans_cost
            portfolio_return = torch.matmul(period_returns, action)
            s_dash = torch.cat((next_sharpes, portfolio_return.view(1)), dim=0)
            state_pairs_list.append(torch.stack((s, s_dash), dim=1))
        
        state_pairs = torch.stack(state_pairs_list, dim=0)
        actions = torch.stack(actions_list, dim=0)
        
        return state_pairs, actions
    
    def calc_log_prob(self, action, mean_vector):
        '''
        Calculates and returns log probability of given action from
        Multivariate Gaussian Distribution w/ parameters mean_vector and preset covariance matrix
            -action: torch tensor of size (num_sel_stocks,)
        
        => Integrates Probability Distribution Function of given Multivariate Gaussian Distribution over
        all possible values of action_temp that could have yielded action (since softmax is a many-to-one function)
        => Note that if (y_1, y_2, ..., y_n) = softmax(x_1, x_2, ..., x_n) then
            x_i = log(y_i) + log(C) for i = 1, 2, ..., n where C = exp(x_1) + exp(x_2) + ... + exp(x_n)
            Thus, x_i = x_1 + log(y_i) - log(y_1) for i >= 2
        '''

        # Multivariate Gaussian PDF
        def pdf(x_1):
            x_elements = [[x_1]]
            for i in range(2, self.num_sel_stocks+1):
                # for expert 0 weights, add epsilon to avoid producing -inf 
                if action[i-1].item() < self.epsilon:
                    x_i = x_1 + torch.log(torch.abs(action[i-1]) + self.epsilon)
                else:
                    x_i = x_1 + torch.log(action[i-1])
                if action[0].item() < self.epsilon:
                    x_i -= torch.log(torch.abs(action[0]) + self.epsilon)
                else:
                    x_i -= torch.log(action[0])
                x_elements.append([x_i.item()])
                
            x = torch.tensor(x_elements, requires_grad=True).to(self.device)
            
            x_m = x - torch.unsqueeze(mean_vector, dim=1)
            
            matmul1 = torch.matmul(x_m.T, self.cov_inv)
            matmul2 = torch.matmul(matmul1, x_m)
            
            return (1.0 / (math.sqrt((2*math.pi)**self.num_sel_stocks*torch.linalg.det(self.cov_mat))) * torch.exp(-matmul2/2)).item()
        
        prob = torch.tensor(integrate.quad(pdf, -np.inf, np.inf)[0], requires_grad=True).to(self.device)
        log_prob = torch.log(prob.view(1))
        
        return log_prob
        
    def _init_hyperparams(self, hyperparameters):

        self.epsilon = 1e-8
        self.actor_init_lr = 0.001
        self.disc_init_lr = 0.001
        self.batch_size = 50
        self.iterations = 1000
        self.rebal_period = 60
        self.num_sel_stocks = 30
        self.alpha = 0.1
        self.trans_cost = 0.003

        self.model_save_path = hyperparameters['model_save_path']
        self.tensorboard_path = hyperparameters['tensorboard_path']

        self.device = hyperparameters['DEVICE']
        
        for param, val in hyperparameters.items():
            if type(val) != str:
                exec('self.' + param + ' = ' + str(val))
                
        assert self.num_sel_stocks < self.rebal_period, 'Rebalancing Period should be greater than Number of Selected Stocks to calculate Max-Sharpe Portfolio'
    