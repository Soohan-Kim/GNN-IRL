from datetime import datetime as dt

mode = 'test' # 'train' or 'test'

all_configs = {
    'train_start' : dt(2002, 1, 1),
    'train_end' : dt(2017, 12, 31),
    'test_start' : dt(2018, 1, 1),
    'test_end' : dt(2021, 12, 31),
    
    'airl_hyperparams' : {
        'epsilon': 5e-3, # for expert 0 weights, add to avoid nan log prob values
        'actor_init_lr': 0.1,
        'disc_init_lr': 0.1,
        'batch_size': 256,
        'iterations': 1000,
        'rebal_period': 60,
        'num_sel_stocks': 3,
        'alpha': 0.1,
        'trans_cost': 0.003,
        'model_save_path': {
            'Actor': './models/actor/sel_3_eps_5e-3_batch_256_lr_0.1', # put only model name (not market, industry or .pt extension)
            'Discriminator': './models/discriminator/sel_3_eps_5e-3_batch_256_lr_0.1' 
        },
        'tensorboard_path': './logs/sel_3_eps_5e-3_batch_256_lr_0.1/' 
    },
    
    'network_hyperparams' : {
        'data_seq_len': 20,
        'LSTM_hiddim': 64,
        'LSTM_attention': True,
        'Disc_gamma': 0.99,
        'g_h_args' : {
            'g_args': {
                'hidden_layers' : [16, 1],
                'size' : 1,
                'activation': 'Identity'
            },
            'h_args': {
                'hidden_layers' : [16, 8, 1],
                'size' : 2,
                'activation' : 'LeakyReLU'
            }
        }
    }
}










    # def rollout(self):
    #     '''
    #     Returns policy-generated state_pairs (s, s')'s, actions a's and log_probs log(pi(a|s))'s
    #         -state_pairs: torch tensor of size (batch size, num_stocks + 1, 2)
    #         -actions: torch tensor of size (batch size, num_stocks) [non selected stocks are 0-filled]
    #         -log_probs: torch tensor of size (batch size, 1)
            
    #     Also updates self.mean_vectors to store current policy mean_vectors per iteration
    #     '''
        
    #     state_pairs_list, actions_list, log_probs_list = [], [], []
    #     self.mean_vectors = []
    #     # For each starting day
    #     for start_id in self.start_ids:
    #         # Get s
    #         x = torch.unsqueeze(self.data[start_id, ...], dim=0)
    #         mean_vector, s_temp = self.actor_net(x, self.A)
    #         s = torch.cat((s_temp, torch.zeros(1, 1).to(self.device)), dim=1)
            
    #         self.mean_vectors.append(torch.squeeze(mean_vector, dim=0))
            
    #         # Get a and log(pi(a|s))
    #         action, log_prob = self.sample_action(torch.squeeze(mean_vector, dim=0), s_temp)
    #         actions_list.append(action)
    #         log_probs_list.append(log_prob)
            
    #         # Get s'
    #         x_dash = torch.unsqueeze(self.data[start_id+1, ...], dim=0)
    #         _, s_dash_temp = self.actor_net(x_dash, self.A)
    #         period_returns = torch.sum(self.raw_data[start_id: start_id + self.rebal_period, :], dim=0) - self.trans_cost
    #         portfolio_return = torch.matmul(period_returns, action)
    #         s_dash = torch.cat((s_dash_temp, portfolio_return.view(1, 1)), dim=1)
            
    #         state_pairs_list.append(torch.stack((s, s_dash), dim=2))
            
    #     state_pairs = torch.cat(state_pairs_list, dim=0)
    #     actions = torch.stack(actions_list, dim=0)
    #     log_probs = torch.stack(log_probs_list, dim=0)
        
    #     return state_pairs, actions, log_probs
    
    # def sample_action(self, mean_vector, cur_scores):
    #     '''
    #     Returns sampled action and log prob given mean vector
    #         -mean_vector: torch tensor of size (num_sel_stocks,)
    #         -cur_scores: torch tensor of size (1, num stocks)
            
    #         -action: torch tensor of size (num_stocks,)
    #         -log_prob: torch tensor of size (1,)
    #     '''
        
    #     dist = MultivariateNormal(loc=mean_vector, covariance_matrix=self.cov_mat)
    #     action_temp = dist.sample()
        
    #     a = nn.functional.softmax(action_temp)
    #     print(a)
    #     log_prob = self.calc_log_prob(a, mean_vector)
        
    #     ###### action 0-filling for non selected stocks ######
    #     _, idx = torch.topk(cur_scores, a.size(dim=-1))
    #     temp_zeroes = torch.zeros_like(cur_scores, requires_grad=True).to(self.device)
    #     action = temp_zeroes.scatter(1, torch.sort(idx)[0], torch.unsqueeze(a, dim=0))
        
    #     return torch.squeeze(action, dim=0), log_prob