import torch
from torch import nn

class LSTM_Net(nn.Module):

  def __init__(self, input_size, hid_dim=64, attention=True):
    super(LSTM_Net, self).__init__()

    self.input_size = input_size
    self.hid_dim = hid_dim
    self.attention = attention

    self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hid_dim, batch_first=True)

    if self.attention:
      self.w = nn.Linear(self.hid_dim, 1)

  def forward(self, x):
    
    out = []
    for i in range(x.size(dim=1)):
      cur_x = x[:, i, :, :]
      cur_out, (h, _) = self.lstm(cur_x)

      if self.attention:
        W = torch.tanh(self.w(cur_out))
        attn_weights = nn.functional.softmax(W, dim=1)
        cur_out = torch.sum(cur_out*attn_weights, dim=1)
      else:
        cur_out = torch.squeeze(h, dim=0)

      out.append(cur_out)
        
    out = torch.stack(out, dim=1)

    return out

class GNN(nn.Module):

  def __init__(self, num_relations):
    super(GNN, self).__init__()

    self.network = nn.Linear(in_features=num_relations, out_features=1)
    self.phi = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, e_old, A):
    num_stocks = e_old.size(dim=1)

    e_new_list = []
    for i in range(num_stocks):
      e_old_i = e_old[:, i, :]
      e_new_i_list = []
      d_j = 0
      for j in range(num_stocks):
        a_ji = A[i, j, :]
        e_old_j = e_old[:, j, :]
        if torch.sum(a_ji).item() > 0:
          d_j += 1
          g = torch.sum(e_old_i * e_old_j, dim=-1) * self.phi(self.network(a_ji)).item()
          # print(g.size()), print(e_old_j.size())
          # break
          g = torch.unsqueeze(g, dim=1)
          e_new_i_list.append(g * e_old_j)
      
      e_new_i = torch.sum(torch.stack(e_new_i_list, dim=0), dim=0) / d_j
      e_new_list.append(e_new_i)

    e_new = torch.stack(e_new_list, dim=1)
    e_final = torch.cat([e_old, e_new], dim=-1)

    return e_final


############### DRIVER MODEL #################

class TGC(nn.Module):
  
  def __init__(self, input_size, num_relations, device, hid_dim=64, attention=True):
    super(TGC, self).__init__()

    self.LSTM = LSTM_Net(input_size, hid_dim, attention).to(device)
    self.GNN = GNN(num_relations).to(device)
    self.pred_layer = nn.Linear(in_features=hid_dim*2, out_features=1).to(device)

  def forward(self, x, A):
    e_old = self.LSTM(x)
    e_final = self.GNN(e_old, A)
    preds = self.pred_layer(e_final)

    return preds


############### RL Actor Class ######################

class Actor(nn.Module):

  def __init__(self, num_stocks, num_sel_stocks, input_size, num_relations, device, hid_dim=64, attention=True):
    super(Actor, self).__init__()

    self.device = device
    self.num_sel_stocks = num_sel_stocks
    self.TGC = TGC(input_size, num_relations, device, hid_dim, attention).to(device)
    self.final_layer = nn.Linear(in_features=num_stocks, out_features=num_sel_stocks).to(device)

  def forward(self, x, A):
    stock_scores = torch.squeeze(self.TGC(x, A), dim=-1)
    top_vals, top_indices = torch.topk(stock_scores, self.num_sel_stocks)
    mask = torch.zeros_like(stock_scores, requires_grad=True).to(self.device)
    final_in = mask.scatter(1, top_indices, top_vals) 
    mean_vector = self.final_layer(final_in)

    return mean_vector, stock_scores


############### MLP Helper Function ###################
def mlp(x,
        hidden_layers,
        activation='Tanh',
        size=2,
        output_activation=nn.Identity):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)

        # For discriminator
        if activation == 'ReLU':
            net_layers.append(nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            net_layers.append(nn.LeakyReLU(.2, inplace=True))
        elif activation == 'Tanh':
            net_layers.append(nn.Tanh())
        else:
            net_layers.append(nn.Identity())
        x = size

    net_layers.append(nn.Linear(x, hidden_layers[-1]))
    net_layers += [output_activation()]

    return nn.Sequential(*net_layers)


############### RL Discriminator Network ################

class Discriminator(nn.Module):
  
  '''
  D(s, a, s') = exp(r(s, a, s')) / {exp(r(s, a, s')) + pi(a|s)}
  '''

  def __init__(self, device, num_stocks, gamma=0.99, **args):
    super(Discriminator, self).__init__()

    self.gamma = gamma

    self.g = mlp(num_stocks, **args['g_args']).to(device)
    self.h = mlp(num_stocks + 1, **args['h_args']).to(device)
    self.sigmoid = nn.Sigmoid()
    self.device = device

  def estimate_reward(self, data, a_e):
    '''
    r(s, a, s') = g(s, a_e) + gamma * h(s') - h(s) [ADVANTAGE ESTIMATE]
    [a_e: action vector extended by filling 0's on positions of not selected stocks]

    data: batch pairs of (s, s') [TENSOR OF SIZE (batch size, num stocks + 1, 2)]
    a_e: action [TENSOR OF SIZE (batch size, num stocks)]

    *batch size: number of trajectories, i.e. (s, a, s') pairs collected per iteration
    '''
    
    s, s_prime = data[:, :, 0], data[:, :, 1]
    
    cur_scores = s[:, :-1]
    
    g_s = self.g(cur_scores*a_e)
    h_s_prime = self.h(s_prime)
    h_s = self.h(s)

    r = g_s + self.gamma * h_s_prime - h_s

    return r # of size (batch size, 1)

  def forward(self, log_p, data, a_e):
    '''
    log_p: batch of pi(a|s) [TENSOR OF SIZE (batch size, 1)]
    '''
    
    adv_est = self.estimate_reward(data, a_e)
    # print('REWARD ###########')
    # print(adv_est)
    exp_adv = torch.exp(adv_est)
    # print('REWARD EXP ###########')
    # print(exp_adv)

    D_value = torch.div(exp_adv, (exp_adv + torch.exp(log_p) + torch.full((log_p.size(dim=0), 1), 1e-8).to(self.device)))
    #print(exp_adv / (exp_adv + torch.exp(log_p)))
    # print('PROBS #################')
    # print(torch.exp(log_p))
    # print('D VALUE ##################')
    # print(D_value)

    return self.sigmoid(D_value)
