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

