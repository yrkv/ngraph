
from functools import partial

import torch
import torch.nn as nn
import numpy as np

import wandb
from tqdm.auto import tqdm

from model import NeuralGraph
from message import *
from update import *
import data


config = {
    "batch_size": 32,
    "word_embed_size": 32,
    "channels_v": 64,
    "channels_e": 64,
    "lr": 0.001,
    "epochs": 50,
    "node_count": 20,
    "start_node_a": 0.2,
    "start_edge_a": 0.2,
    "message_config": 1,
    "message_zero": False,
    "update_config": 0,
    "update_zero": False,
    "reset_out_node": True,
    "step_edges": True,
    "node_dropout_p": 0.0,
    "edge_dropout_p": 0.0,
    "add_nodes": 0,
    "add_node_every": 1,
    #"load_ckpt": None,
    "load_ckpt": 'ngraph20_e15.pt',
    #"load_ckpt": 'learngen20_e15.pt',
    #"load_ckpt": 'learn20_e10.pt',
    #"save_ckpt": None,

    # Do graphs share the same trainable base values?
    'shared_base': True,

    'persist': {
        # epochs to run with persist.fraction=0
        'epoch_start': 0,
        # epochs at which to have full persist.fraction
        'epoch_end': 0,
        # fraction of graphs to reset before each batch based on loss
        'fraction': 0.0,
    }
}



# task config
# BATCH_TOKENS = 35
BATCH_TOKENS = 60
VOCAB_SIZE = 26
#LANG_STAGES = [50, 50, 50, 50, VOCAB_SIZE]
#LANG_STAGES = [500, 500, 500, 500, 500, 500, VOCAB_SIZE]
LANG_STAGES = [50, 50, VOCAB_SIZE]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TUNE_INITIAL_VALUES = True
FREEZE_MODEL = False

INPUT_NODE = 0
OUTPUT_NODE = 1


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
        # zero output node for consistent start point for each token
        if wandb.config['reset_out_node']:
            ngraph.node_vals[:, OUTPUT_NODE, :emb.embedding_dim] = 0

        #ngraph.node_vals.clamp_(-10, 10)
        #ngraph.edge_vals.clamp_(-10, 10)

        for cycle_i in range(len(ngraph.message)):
            # input read tokens one by one
            ngraph.node_vals[:, INPUT_NODE, :emb.embedding_dim] = emb(x)

            # label nodes in last channels
            ngraph.node_vals[:, :, -1] = 0
            ngraph.node_vals[:, INPUT_NODE, -1] = 1
            ngraph.node_vals[:, :, -2] = 0
            ngraph.node_vals[:, OUTPUT_NODE, -2] = 1

            ngraph.timestep(nodes=True, edges=False, layer=cycle_i)
        if wandb.config['step_edges']:
            ngraph.timestep(nodes=False, edges=True)

        out.append((ngraph.node_vals[:, OUTPUT_NODE, :emb.embedding_dim] + 0.))
    
    pred = decoder(torch.stack(out))
    return pred.permute(1,0,2)


def train_epoch(ngraph, base_node, base_edge, emb, decoder,
                optimizer, criterion, train_data, epoch):
    indices = list(range(0, train_data.size(1) - 1, BATCH_TOKENS))
    np.random.shuffle(indices)
    bar = tqdm(indices)

    for i in bar:
        x_data = train_data[:, i:i+BATCH_TOKENS].to(DEVICE)
        y_data = train_data[:, i+1:i+1+BATCH_TOKENS].to(DEVICE)
        if x_data.shape != y_data.shape:
            continue
        
        fraction = wandb.config['persist.fraction']
        start, end = wandb.config['persist.epoch_start'], wandb.config['persist.epoch_end']
        actual_fraction = np.clip((epoch-start+1) / (end-start+1) * fraction, 0, fraction)
        reset_above = np.quantile(ngraph.scores, 1-actual_fraction)
        reset = torch.from_numpy(ngraph.scores > reset_above).to(DEVICE)
        reset = reset.reshape(-1, 1, 1)
        reset.requires_grad_(False)
        
        if wandb.config['shared_base']:
            base_n, base_e = base_node[:1].expand(32, -1, -1),  base_edge[:1].expand(32, -1, -1)
        else:
            base_n, base_e = base_node, base_edge
            
        ngraph.node_vals = base_n * reset + ngraph.node_vals.detach() * (~reset)
        ngraph.edge_vals = base_e * reset + ngraph.edge_vals.detach() * (~reset)
        
        #ngraph.node_vals += torch.randn_like(ngraph.node_vals) * 0.05
        #ngraph.edge_vals += torch.randn_like(ngraph.edge_vals) * 0.05

        pred = process_input(ngraph, emb, decoder, x_data)

        if base_node.abs().max() > 100:
            print(f'high base_node vals! {base_node.abs().max()}')
        if base_edge.abs().max() > 100:
            print(f'high base_edge vals! {base_edge.abs().max()}')
        #if ngraph.node_vals.abs().max() > 100:
        #    print(f'high node vals! {ngraph.node_vals.abs().max()}')
        #if ngraph.edge_vals.abs().max() > 100:
        #    print(f'high edge vals! {ngraph.edge_vals.abs().max()}')

        losses = nn.CrossEntropyLoss(reduction='none')(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
        losses = losses.reshape(y_data.shape).mean(1)
        ngraph.scores = losses.detach().cpu().numpy()

        loss = losses.mean()
        #overflow = ngraph.overflow(10)
        #if epoch == 0:
            #overflow = loss * 0.

        #if overflow > 1:
            #loss = loss * 0
        full_loss = loss# + overflow

        # skip optimization if we have inf or nan
        if full_loss.isnan() or full_loss.isinf():
            print('INF or NAN detected!!!')
            optimizer.zero_grad()
            full_loss.backward()
            assert False


        full_loss.backward()

        torch.nn.utils.clip_grad_norm_(params, 0.5)

        optimizer.step()


        entry = {
                'loss': loss.item(),
                #'overflow': overflow.item(),
                }
        bar.set_postfix(entry)
        wandb.log(entry)

    #assert False


def evaluate(ngraph, base_node, base_edge, emb, decoder, val_data):
    with torch.no_grad():
        indices = list(range(0, val_data.size(1) - 1, BATCH_TOKENS))
        np.random.shuffle(indices)
        bar = tqdm(indices)

        total_count = 0
        total_loss = 0

        for i in bar:
            x_data = train_data[:, i:i+BATCH_TOKENS].to(DEVICE)
            y_data = train_data[:, i+1:i+1+BATCH_TOKENS].to(DEVICE)
            if x_data.shape != y_data.shape:
                continue


            if wandb.config['shared_base']:
                ngraph.node_vals = base_node[:1].expand(32, -1, -1) + 0.
                ngraph.edge_vals = base_edge[:1].expand(32, -1, -1) + 0.
            else:
                ngraph.node_vals = base_node + 0.
                ngraph.edge_vals = base_edge + 0.

            pred = process_input(ngraph, emb, decoder, x_data)

            loss = criterion(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
            total_loss += loss.item()
            total_count += 1
            bar.set_postfix({'val_loss': total_loss / total_count})

        wandb.log({'val_loss': total_loss / total_count})


def evaluate_learning(ngraph, base_node, base_edge, emb, decoder, learn_data):
    with torch.no_grad():
        node_vals_pre = ngraph.node_vals
        edge_vals_pre = ngraph.edge_vals

        if wandb.config['shared_base']:
            ngraph.node_vals = base_node[:1].expand(32, -1, -1) + 0.
            ngraph.edge_vals = base_edge[:1].expand(32, -1, -1) + 0.
        else:
            ngraph.node_vals = base_node.flip(0)
            ngraph.edge_vals = base_edge.flip(0)

        indices = list(range(0, learn_data.size(1) - 1, BATCH_TOKENS))
        np.random.shuffle(indices)
        bar = tqdm(indices)
        for i in bar:
            x_data = train_data[:, i:i+BATCH_TOKENS].to(DEVICE)
            y_data = train_data[:, i+1:i+1+BATCH_TOKENS].to(DEVICE)
            if x_data.shape != y_data.shape:
                continue

            pred = process_input(ngraph, emb, decoder, x_data)
            loss = criterion(pred.reshape(-1, VOCAB_SIZE), y_data.reshape(-1))
            bar.set_postfix({'learn_val_loss': loss.item()})
            wandb.log({'learn_val_loss': loss.item()})

        ngraph.node_vals = node_vals_pre
        ngraph.edge_vals = edge_vals_pre


#from torchtext.datasets import WikiText2
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
#
#from torch import Tensor
#from torch.utils.data import dataset
#
#train_iter = WikiText2(split='train')
#tokenizer = get_tokenizer('basic_english')
#vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
#vocab.set_default_index(vocab['<unk>'])
#
#VOCAB_SIZE = len(vocab)
#
#def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
#    """Converts raw text into a flat Tensor."""
#    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
#    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
#
## ``train_iter`` was "consumed" by the process of building the vocab,
## so we have to create it again
#train_iter, val_iter, test_iter = WikiText2()
#train_data = data_process(train_iter)
#val_data = data_process(val_iter)
#test_data = data_process(test_iter)
#
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#def batchify(data: Tensor, bsz: int) -> Tensor:
#    """Divides the data into ``bsz`` separate sequences, removing extra elements
#    that wouldn't cleanly fit.
#
#    Arguments:
#        data: Tensor, shape ``[N]``
#        bsz: int, batch size
#
#    Returns:
#        Tensor of shape ``[N // bsz, bsz]``
#    """
#    seq_len = data.size(0) // bsz
#    data = data[:seq_len * bsz]
#    #data = data.view(bsz, seq_len).t().contiguous()
#    data = data.view(bsz, seq_len).contiguous()
#    return data.to(device)
#
#batch_size = 32
#eval_batch_size = 32
#train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
#val_data = batchify(val_data, eval_batch_size)
#test_data = batchify(test_data, eval_batch_size)


if __name__ == '__main__':

    run = wandb.init(config=config, project='ngraph')
    print(wandb.config)

    assert False
    
    lr = wandb.config['lr']
    bs = wandb.config['batch_size']
    epochs = wandb.config['epochs']

    ch_v = wandb.config['channels_v']
    ch_e = wandb.config['channels_e']

    # lang = data.Language(LANG_STAGES)
    langs = [data.Language(LANG_STAGES) for _ in range(bs)]
    learn_langs = [data.Language(LANG_STAGES) for _ in range(bs)]

    print('generating data...')
    train_data = generate_batched_data(langs, bs, base_len=1000, progress=True)#.to(DEVICE)
    test_data = generate_batched_data(langs, bs, base_len=100)#.to(DEVICE)
    learn_data = generate_batched_data(langs, bs, base_len=100)#.to(DEVICE)
    print(f'{train_data.shape=}\n{test_data.shape=}')
    #assert False

    message_functions = [
        partial(message_tiny, ch_v, ch_e, zero=wandb.config['message_zero']),
        partial(message_tiny_plus, ch_v, ch_e, zero=wandb.config['message_zero']),
        partial(message_small, ch_v, ch_e, 16, zero=wandb.config['message_zero']),
        partial(message_small_plus, ch_v, ch_e, 16, zero=wandb.config['message_zero']),
    ]
    message_f = message_functions[wandb.config['message_config']]

    update_functions = [
        partial(update_tiny, ch_v, zero=wandb.config['update_zero']),
        partial(update_sigmoid_tiny, ch_v, zero=wandb.config['update_zero']),
        partial(update_small, ch_v, 16, zero=wandb.config['update_zero']),
    ]
    update_f = update_functions[wandb.config['update_config']]

    ngraph = NeuralGraph(wandb.config['node_count'], message_f, update_f,
                         ch_v=ch_v, ch_e=ch_e,
                         node_dropout_p=wandb.config['node_dropout_p'], 
                         edge_dropout_p=wandb.config['edge_dropout_p'], 
                         batchsize=bs,
                         average_messages=True,
                         layers=3).to(DEVICE)

    emb = nn.Embedding(VOCAB_SIZE, wandb.config['word_embed_size']).to(DEVICE)
    decoder = nn.Linear(wandb.config['word_embed_size'], VOCAB_SIZE).to(DEVICE)

    if wandb.config['load_ckpt'] is not None:
        loaded = torch.load(wandb.config['load_ckpt'])

        ngraph = loaded['ngraph'].to(DEVICE)
        emb = loaded['emb'].to(DEVICE)
        decoder = loaded['decoder'].to(DEVICE)


    if FREEZE_MODEL:
        ngraph.message.requires_grad_(False)
        ngraph.update.requires_grad_(False)

    ngraph.reset_values(edge_a=wandb.config['start_edge_a'],
                        node_a=wandb.config['start_node_a'])

    base_node = ngraph.node_vals.clone().detach().to(DEVICE)
    base_edge = ngraph.edge_vals.clone().detach().to(DEVICE)


    if TUNE_INITIAL_VALUES:
        base_node.requires_grad_(True)
        base_edge.requires_grad_(True)


    params = [*ngraph.parameters(), *emb.parameters(), *decoder.parameters()]
    if TUNE_INITIAL_VALUES:
        params.extend([base_node, base_edge])

    optimizer = torch.optim.AdamW(params, lr=lr)
    #optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        #k = wandb.config['add_node_every']
        #if epoch % k == 0 and epoch > 0 and epoch // k <= wandb.config['add_nodes']:
        #    ngraph.node_vals = base_node + 0.
        #    ngraph.edge_vals = base_edge + 0.
        #    ngraph.add_node()
        #    base_node = ngraph.node_vals.clone().detach()
        #    base_edge = ngraph.edge_vals.clone().detach()
        #    base_node.requires_grad_(True)
        #    base_edge.requires_grad_(True)
        #    optimizer.add_param_group({'params': [base_node, base_edge]})

        #if epoch == 25:
        #    ngraph.message.requires_grad_(True)
        #    ngraph.update.requires_grad_(True)

        train_epoch(ngraph, base_node, base_edge, emb, decoder,
                    optimizer, criterion, train_data, epoch)

            # assert False
        evaluate(ngraph, base_node, base_edge, emb, decoder, test_data)
        evaluate_learning(ngraph, base_node, base_edge, emb, decoder, learn_data)
        #torch.save(ngraph.to('cpu'), 'ngraph30_high_ch.pt')
        if (epoch+1) % 5 == 0:
            #torch.save({
            #    'ngraph': ngraph.to('cpu'),
            #    'emb': emb.to('cpu'),
            #    'decoder': decoder.to('cpu'),
            #}, f'learngen20_e{epoch+1}.pt')
            ngraph.to(DEVICE)
            emb.to(DEVICE)
            decoder.to(DEVICE)












