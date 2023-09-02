
from functools import partial
import argparse

import torch
import torch.nn as nn
import numpy as np

import wandb
from tqdm.auto import tqdm

from NeuralGraph import NeuralGraph
from message import *
from update import *
from attention import *
import data


def get_config():
    mbool = lambda x: (str(x).lower() in ['true','1', 'yes'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_embed_size', type=int, default=32)
    parser.add_argument('--channels_n', type=int, default=64)
    parser.add_argument('--channels_e', type=int, default=64)
    parser.add_argument('--channels_k', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--node_count', type=int, default=20)
    # parser.add_argument('--start_node_a', type=float, default=0.2)
    # parser.add_argument('--start_edge_a', type=float, default=0.2)
    parser.add_argument('--init_value_std', type=float, default=0.2)

    parser.add_argument('--message_config', type=int, default=1)
    parser.add_argument('--message_zero', type=mbool, default=False)

    parser.add_argument('--update_config', type=int, default=0)
    parser.add_argument('--update_zero', type=mbool, default=False)

    parser.add_argument('--attention_config', type=int, default=1)
    parser.add_argument('--attention_zero', type=mbool, default=False)

    # parser.add_argument('--reset_out_node', type=mbool, default=True)
    # parser.add_argument('--step_edges', type=mbool, default=True)

    parser.add_argument('--node_dropout_p', type=float, default=0.0)
    parser.add_argument('--edge_dropout_p', type=float, default=0.0)

    # parser.add_argument('--add_nodes', type=int, default=0)
    # parser.add_argument('--add_node_every', type=int, default=1)

    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--save_ckpt', type=str, default=None)

    parser.add_argument('--optimize_model', type=mbool, default=True)
    parser.add_argument('--optimize_base', type=mbool, default=True)

    parser.add_argument('--shared_base', type=mbool, default=False,
                        help="do graphs share the same trainable base values?")
    # parser.add_argument('--persist.epoch_start', type=int, default=0,
    #                     help="epochs to run with persist.reset_fraction=0")
    # parser.add_argument('--persist.epoch_end', type=int, default=0,
    #                     help="epochs at which to have full persist.fraction")
    # parser.add_argument('--persist.reset_fraction', type=float, default=1,
    #                     help="fraction of graphs to reset before each batch based on loss")
    args, _ = parser.parse_known_args()

    def fix_nesting(d: dict, key, value):
        i = key.find('.')
        if i == -1:
            d[key] = value
            return
        if key[:i] not in d:
            d[key[:i]] = {}
        fix_nesting(d[key[:i]], key[i+1:], value)

    # convert keys with "." in them into a nested dict inside
    config = {}
    for k, v in args.__dict__.items():
        fix_nesting(config, k, v)

    return config


# task config
BATCH_TOKENS = 100
VOCAB_SIZE = 26
#LANG_STAGES = [50, 50, 50, 50, VOCAB_SIZE]
#LANG_STAGES = [500, 500, 500, 500, 500, 500, VOCAB_SIZE]
LANG_STAGES = [50, 50, VOCAB_SIZE]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_NODE = 0
OUTPUT_NODE = -1


def generate_batched_data(langs, batch_size, base_len=1_000, progress=False):
    if type(langs) is not list:
        langs = [langs]*batch_size
    bar = tqdm(langs) if progress else langs
    sequences = [lang.generate(base_len) for lang in bar]
    min_len = min(map(len, sequences))
    arr = np.array([seq[:min_len] for seq in sequences], dtype='int64')
    return torch.from_numpy(arr)


def process_input(ngraph, emb, decoder, X):
    out = []
    for x in X.T:
        inp = emb(x).unsqueeze(-2)
        #ngraph.nodes[:, INPUT_NODE, :wandb.config['word_embed_size']] += emb(x)
        #ngraph.nodes[:, INPUT_NODE, :wandb.config['word_embed_size']] = emb(x)
        #ngraph.apply_vals(inp)

        for t in range(ngraph.n_models):
            ngraph.nodes[:, INPUT_NODE, :wandb.config['word_embed_size']] += emb(x)
            #ngraph.nodes[:, INPUT_NODE, :wandb.config['word_embed_size']] = emb(x)
            #ngraph.apply_vals(inp)
            
            ngraph.nodes[:, :, -1] = 0
            ngraph.nodes[:, INPUT_NODE, -1] = 1
            ngraph.nodes[:, :, -2] = 0
            ngraph.nodes[:, OUTPUT_NODE, -2] = 1

            ngraph.timestep(step_nodes=True, step_edges=True, dt=1, t=t)
            #ngraph.timestep(nodes=True, edges=False, dt=1, t=t)

        #ngraph.timestep(nodes=True, edges=True, dt=1, t=0)
        #ngraph.timestep(step_nodes=True, step_edges=True, dt=1, t=0)

        # ngraph.forward(inp, time=5, dt=1, nodes=True, edges=True)
        
        #out.append((ngraph.nodes[:, OUTPUT_NODE, :emb.embedding_dim] + 0.))
        out.append(ngraph.read_outputs())

    pred = decoder(torch.stack(out))
    return pred.squeeze(2).permute(1,0,2)


def train_epoch(ngraph, emb, decoder, optimizer, train_data, epoch):
    indices = list(range(0, train_data.size(1) - 1, BATCH_TOKENS))
    np.random.shuffle(indices)
    bar = tqdm(indices)

    for i in bar:
        selection = np.random.choice(len(train_data), wandb.config['batch_size'], replace=False)
        x_data = train_data[selection, i:i+BATCH_TOKENS].to(DEVICE)
        y_data = train_data[selection, i+1:i+1+BATCH_TOKENS].to(DEVICE)
        if x_data.shape != y_data.shape:
            continue

        ngraph.init_vals(batch_size=wandb.config['batch_size'])
        pred = process_input(ngraph, emb, decoder, x_data)

        losses = nn.CrossEntropyLoss(reduction='none')(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
        losses = losses.reshape(y_data.shape).mean(1)

        task_loss = losses.mean()
        overflow = ngraph.overflow(10)
        loss = task_loss + overflow

        # bail if we have inf or nan
        if loss.isnan() or loss.isinf():
            assert False, 'INF or NAN detected!!!'

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*ngraph.parameters(), *emb.parameters(), *decoder.parameters()], 1.0)
        optimizer.step()

        entry = {
            'loss': task_loss.item(),
            'overflow': overflow.item(),
        }
        bar.set_postfix(entry)
        wandb.log(entry)


def evaluate(ngraph, emb, decoder, val_data, log_key='val_loss'):
    with torch.no_grad():
        indices = list(range(0, val_data.size(1) - 1, BATCH_TOKENS))
        np.random.shuffle(indices)
        bar = tqdm(indices)

        total_count = 0
        total_loss = 0

        for i in bar:
            x_data = val_data[:, i:i+BATCH_TOKENS].to(DEVICE)
            y_data = val_data[:, i+1:i+1+BATCH_TOKENS].to(DEVICE)
            if x_data.shape != y_data.shape:
                continue

            ngraph.init_vals(batch_size=wandb.config['batch_size'])
            pred = process_input(ngraph, emb, decoder, x_data)

            loss = nn.CrossEntropyLoss()(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
            total_loss += loss.item()
            total_count += 1
            bar.set_postfix({log_key: total_loss / total_count})

        wandb.log({log_key: total_loss / total_count})


# def evaluate_learning(ngraph, emb, decoder, learn_data, log_key='learn_loss'):
#     with torch.no_grad():
#         node_vals_pre = ngraph.node_vals
#         edge_vals_pre = ngraph.edge_vals

#         if wandb.config['shared_base']:
#             ngraph.node_vals = base_node[:1].expand(wandb.config['batch_size'], -1, -1) + 0.
#             ngraph.edge_vals = base_edge[:1].expand(wandb.config['batch_size'], -1, -1) + 0.
#         else:
#             ngraph.node_vals = base_node.flip(0)
#             ngraph.edge_vals = base_edge.flip(0)

#         indices = list(range(0, learn_data.size(1) - 1, BATCH_TOKENS))
#         np.random.shuffle(indices)
#         bar = tqdm(indices)
#         for i in bar:
#             x_data = learn_data[:, i:i+BATCH_TOKENS].to(DEVICE)
#             y_data = learn_data[:, i+1:i+1+BATCH_TOKENS].to(DEVICE)
#             if x_data.shape != y_data.shape:
#                 continue

#             pred = process_input(ngraph, emb, decoder, x_data)
#             loss = nn.CrossEntropyLoss()(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
#             bar.set_postfix({log_key: loss.item()})
#             wandb.log({log_key: loss.item()})

#         ngraph.node_vals = node_vals_pre
#         ngraph.edge_vals = edge_vals_pre



if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    run = wandb.init(config=get_config(), project='ngraph')

    print(wandb.config)

    lr = wandb.config['lr']
    bs = wandb.config['batch_size']
    epochs = wandb.config['epochs']

    ch_n = wandb.config['channels_n']
    ch_e = wandb.config['channels_e']
    ch_k = wandb.config['channels_k']

    langs = [data.Language(LANG_STAGES) for _ in range(bs*20)]
    learn_langs = [data.Language(LANG_STAGES) for _ in range(bs)]

    print('generating data...')
    train_data = generate_batched_data(langs, len(langs), base_len=10_000, progress=True)
    test_data = generate_batched_data(langs[:bs], bs, base_len=1_000)
    learn_data = generate_batched_data(learn_langs, bs, base_len=1_000)
    print(f'{train_data.shape=}\n{test_data.shape=}')

    message_functions = [
        partial(message_tiny, ch_extra=0, zero=wandb.config['message_zero']),
        partial(message_tiny_plus, ch_extra=0, zero=wandb.config['message_zero']),
        partial(message_small, hidden=16, ch_extra=0, zero=wandb.config['message_zero']),
        partial(message_small_plus, hidden=16, ch_extra=0, zero=wandb.config['message_zero']),
    ]
    message_f = message_functions[wandb.config['message_config']]

    update_functions = [
        partial(update_tiny, ch_extra=0, zero=wandb.config['update_zero']),
        partial(update_tiny_pre, ch_extra=0, zero=wandb.config['update_zero']),
        partial(update_tiny_post, ch_extra=0, zero=wandb.config['update_zero']),
        partial(update_small, hidden=16, ch_extra=0, zero=wandb.config['update_zero']),
    ]
    update_f = update_functions[wandb.config['update_config']]

    attention_functions = [
        None, # No attention should also be an option
        partial(attention_tiny, ch_extra=0, zero=wandb.config['attention_zero']),
        partial(attention_tiny_plus, ch_extra=0, zero=wandb.config['attention_zero']),
        partial(attention_small, hidden=16, ch_extra=0, zero=wandb.config['attention_zero']),
        partial(attention_small_plus, hidden=16, ch_extra=0, zero=wandb.config['attention_zero']),
    ]
    attention_f = attention_functions[wandb.config['attention_config']]

    # ngraph = NeuralGraph(wandb.config['node_count'], message_f, update_f, attention_function=attention_f,
    #                      ch_v=ch_v, ch_e=ch_e, ch_k=ch_k,
    #                      node_dropout_p=wandb.config['node_dropout_p'], 
    #                      edge_dropout_p=wandb.config['edge_dropout_p'], 
    #                      batchsize=bs,
    #                      average_messages=True,
    #                      layers=3).to(DEVICE)
    class InputIntegrator(nn.Module):
        def __init__(self, ch_inp:int=8, ch_n:int=8):
            super().__init__()

            self.main = nn.Sequential(
                nn.Linear(ch_inp+ch_n, ch_n),
                nn.ReLU(),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.main(x)

    class OutputInterpreter(nn.Module):
        def __init__(self, ch_out:int=8, ch_n:int=8):
            super().__init__()

            self.main = nn.Sequential(
                nn.Linear(ch_n, ch_out),
            )

        def forward(self, x):
            return self.main(x)


    connections = [(x,y) for x in range(wandb.config['node_count']) for y in range(wandb.config['node_count'])]
    ngraph = NeuralGraph(
        wandb.config['node_count'], 1, 1,
        connections=connections,
        ch_n=ch_n, ch_e=ch_e, ch_k=ch_k,
        ch_inp=wandb.config['word_embed_size'],
        ch_out=wandb.config['word_embed_size'],
        decay=0,
        leakage=0,
        #value_init='trainable_batch' if wandb.config['optimize_base'] else 'random',
        value_init='trainable',
        init_value_std=wandb.config['init_value_std'],
        aggregation="attention" if attention_f is not None else "mean",
        device=DEVICE,
        node_dropout_p=wandb.config['node_dropout_p'],
        edge_dropout_p=wandb.config['edge_dropout_p'], 
        n_models=3,
        use_label=False,
        message_generator=message_f,
        update_generator=update_f,
        attention_generator=attention_f,
        inp_int_generator=InputIntegrator,
        out_int_generator=OutputInterpreter,
        clamp_mode='soft',
        max_value=1e6,
    )
    ngraph.init_vals(batch_size=wandb.config['batch_size'])

    #ngraph.inp_int.requires_grad_(False)
    #ngraph.out_int.requires_grad_(False)


    print(ngraph.ch_n, ngraph.ch_e, ngraph.ch_k)#, ngraph.ch_extra)

    # ngraph.reset_values(edge_a=wandb.config['start_edge_a'],
    #                     node_a=wandb.config['start_node_a'])

    emb = nn.Embedding(VOCAB_SIZE, wandb.config['word_embed_size']).to(DEVICE)
    decoder = nn.Linear(wandb.config['word_embed_size'], VOCAB_SIZE).to(DEVICE)

    # # when loading from a checkpoint, ngraph configuration (e.g. node count) is
    # # effectively ignored since the ngaph is loaded directly. 
    if wandb.config['load_ckpt'] is not None:
        loaded = torch.load(wandb.config['load_ckpt'])

        #init_nodes = loaded['ngraph'].pop('init_nodes')
        #init_edges = loaded['ngraph'].pop('init_edges')
        #ngraph.load_state_dict(loaded['ngraph'], strict=False)
        ngraph.load_state_dict(loaded['ngraph'])
        emb.load_state_dict(loaded['emb'])
        decoder.load_state_dict(loaded['decoder'])

        #with torch.no_grad():
            #ngraph.init_nodes.set_(init_nodes[0])
            #ngraph.init_edges.set_(init_edges[0])

    if not wandb.config['optimize_model']:
        ngraph.messages.requires_grad_(False)
        ngraph.updates.requires_grad_(False)
        if attention_f is not None:
            ngraph.attentions.requires_grad_(False)

    # base_node = ngraph.node_vals.clone().detach().to(DEVICE)
    # base_edge = ngraph.edge_vals.clone().detach().to(DEVICE)

    # if wandb.config['optimize_base']:
    #     base_node.requires_grad_(True)
    #     base_edge.requires_grad_(True)


    # params = [*ngraph.parameters(), *emb.parameters(), *decoder.parameters()]
    # if wandb.config['optimize_base']:
        # params.extend([base_node, base_edge])

    optimizer = torch.optim.AdamW([*ngraph.parameters(), *emb.parameters(), *decoder.parameters()], lr=lr)
    #optimizer = torch.optim.Adam(ngraph.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer = torch.optim.RMSprop(params, lr=lr)

    for epoch in range(epochs):

        train_epoch(ngraph, emb, decoder, optimizer, train_data, epoch)
        evaluate(ngraph, emb, decoder, test_data)
        evaluate(ngraph, emb, decoder, learn_data, log_key="learn_val_loss")

        #evaluate_learning(ngraph, base_node, base_edge, emb, decoder, learn_data)

        if wandb.config['save_ckpt'] is not None:
            torch.save({
                'ngraph': ngraph.state_dict(),
                'emb': emb.state_dict(),
                'decoder': decoder.state_dict(),
            }, wandb.config['save_ckpt'])


