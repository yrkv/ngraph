import torch
import torch.nn as nn

import numpy as np
from tqdm.auto import tqdm



class NeuralGraph(nn.Module):
    def __init__(self, nodes:int, message_function, update_function,
                 n_input:int=0, n_output:int=0, connections:list=None,
                 ch_v:int=8, ch_e:int=8, layers:int=5,
                 node_dropout_p:float=0.0, edge_dropout_p:float=0.0, 
                 zero_last:bool=False,
                 batchsize:int=4, poolsize:int=None, 
                 use_update:bool=True, average_messages:bool=True,):
        super().__init__()
        
        self.nodes = nodes
        self.n_input, self.n_output = n_input, n_output
        self.batchsize = batchsize
        self.poolsize = poolsize or batchsize
        self.ch_v, self.ch_e = ch_v, ch_e
        self.node_dropout_p, self.edge_dropout_p = node_dropout_p, edge_dropout_p
        self.zero_last = zero_last
        self.message = nn.ModuleList([message_function() for _ in range(layers)])
        self.update = nn.ModuleList([update_function() for _ in range(layers)])
        # self.message = message or MessageFunction(ch_v=ch_v, ch_e=ch_e)
        # self.update = update or UpdateFunction(ch_v=ch_v)
        self.use_update, self.average_messages = use_update, average_messages
        
        if connections is None:
            connections = [(x,y) for x in range(nodes) for y in range(nodes)]
        
        self.edges = len(connections)
        
        conn_a, conn_b = zip(*connections)
        self.register_buffer('conn_a', torch.tensor(conn_a).long())
        self.register_buffer('conn_b', torch.tensor(conn_b).long())
        
        self.register_buffer('node_vals', torch.zeros(self.poolsize, self.nodes, ch_v))
        self.register_buffer('edge_vals', torch.zeros(self.poolsize, self.edges, ch_e))
        self.reset_values()

        self.register_buffer('counts_a',torch.zeros(nodes).long())
        self.register_buffer('counts_b',torch.zeros(nodes).long())
        self.counts_a.index_add_(0, *torch.unique(self.conn_a, return_counts=True))
        self.counts_b.index_add_(0, *torch.unique(self.conn_b, return_counts=True))
        
        self.register_buffer('counts_ab', self.counts_a + self.counts_b)
        
        # avoid dividing by zero
        self.counts_a[self.counts_a == 0] = 1
        self.counts_b[self.counts_b == 0] = 1
        self.counts_ab[self.counts_ab == 0] = 1
        
        self.pool = np.arange(self.batchsize)
        self.scores = np.ones(self.poolsize)  # high initial scores avoids resetting them initially
    
    
    def timestep(self, nodes=True, edges=True, layer=0):
        zero_nodes = torch.rand(1, self.nodes, 1, device=self.node_vals.device)
        zero_edges = torch.rand(1, self.edges, 1, device=self.edge_vals.device)
        nodes_dropped = self.node_vals * (zero_nodes > self.node_dropout_p)
        edges_dropped = self.edge_vals * (zero_edges > self.edge_dropout_p)
        h_a = nodes_dropped[self.pool[:, None], self.conn_a]
        h_b = nodes_dropped[self.pool[:, None], self.conn_b]
        h_ab = edges_dropped[self.pool]
        
        h = torch.cat([h_a, h_b, h_ab], dim=-1)
        m = self.message[layer](h)
        m_a, m_b, m_ab = torch.split(m, [self.ch_v, self.ch_v, self.ch_e], -1)
        
        agg_m_a = torch.zeros(self.batchsize, self.nodes, self.ch_v, device=h.device)
        agg_m_b = torch.zeros(self.batchsize, self.nodes, self.ch_v, device=h.device)

        agg_m_a.index_add_(1, self.conn_a, m_a)
        agg_m_b.index_add_(1, self.conn_b, m_b)
            
        # Use the vertex update function on aggregated messages
        if self.use_update:
            # Average messages as opposed to summing 
            if self.average_messages:
                agg_m_a.divide_(self.counts_a[None, :, None])
                agg_m_b.divide_(self.counts_b[None, :, None])
            
            m = torch.cat([agg_m_a, agg_m_b, self.node_vals[self.pool]], axis=2)
            updates = self.update[layer](m)
        
        # Use simple average of aggregated messages
        else:
            updates = agg_m_a + agg_m_b
            if self.average_messages:
                updates.divide_(self.counts_ab[None, :, None])
            
        if nodes: self.node_vals[self.pool] += updates
        if edges: self.edge_vals[self.pool] += m_ab
        
        
    def learn(self, X, Y, steps=5, progress=False, reset_nodes=True):
        for x, y in zip(tqdm(X) if progress else X, Y):
            if reset_nodes:
                self.reset_values(nodes=True, edges=False)

            for i in range(steps):
                self.apply_values(input=x.flatten(1), output=y.flatten(1))
                self.timestep(nodes=True, edges=False, layer=i)

            self.timestep(nodes=True, edges=True)
        
        
    def predict(self, X, steps=5, progress=False, reset_nodes=True):
        pred = []
        for x in (tqdm(X) if progress else X):
            if reset_nodes:
                self.reset_values(nodes=True, edges=False)

            for i in range(steps):
                self.apply_values(input=x.flatten(1), output=None)
                self.timestep(nodes=True, edges=False, layer=i)

            pred.append(self.read_outputs() + 0.)
        return torch.stack(pred)
    
    
    def apply_values(self, input, output=None):
        self.node_vals[self.pool, 0:self.n_input, 0] = input
        
        if output is not None:
            self.node_vals[self.pool, -self.n_output:, 0] = output
    
    
    def read_outputs(self):
        return self.node_vals[self.pool, -self.n_output:, 0]
    
    
    def reset_values(self, nodes=True, edges=True, index=slice(None), edge_a=0.1, node_a=0.01):
        if nodes: self.node_vals[index] = torch.randn_like(self.node_vals[index]) * node_a
        if edges: self.edge_vals[index] = torch.randn_like(self.edge_vals[index]) * edge_a
    
    
    def overflow(self, k=5):
        node, edge = self.node_vals[self.pool], self.edge_vals[self.pool]
        node_overflow = ((node - node.clamp(-k, k))**2).mean()
        edge_overflow = ((edge - edge.clamp(-k, k))**2).mean()
        return node_overflow + edge_overflow

    
    def add_node(self, node_a=0.01, edge_a=0.01):
        dev = self.node_vals.device
        add_conn = [
            *((a, self.nodes) for a in range(self.nodes)),
            *((self.nodes, b) for b in range(self.nodes+1)),
        ]
        self.nodes += 1
        self.edges += len(add_conn)

        add_conn_a, add_conn_b = zip(*add_conn)
        add_conn_a = torch.tensor(add_conn_a, device=self.conn_a.device)
        add_conn_b = torch.tensor(add_conn_b, device=self.conn_b.device)
        self.register_buffer('conn_a', torch.cat([self.conn_a, add_conn_a]))
        self.register_buffer('conn_b', torch.cat([self.conn_b, add_conn_b]))

        add_node_vals = torch.randn(self.poolsize, 1, self.ch_v, device=dev) * node_a
        add_edge_vals = torch.randn(self.poolsize, len(add_conn), self.ch_e, device=dev) * edge_a
        self.register_buffer('node_vals', torch.cat([self.node_vals, add_node_vals], dim=1))
        self.register_buffer('edge_vals', torch.cat([self.edge_vals, add_edge_vals], dim=1))

        self.register_buffer('counts_a', torch.zeros(self.nodes, device=dev).long())
        self.register_buffer('counts_b', torch.zeros(self.nodes, device=dev).long())
        self.counts_a.index_add_(0, *torch.unique(self.conn_a, return_counts=True))
        self.counts_b.index_add_(0, *torch.unique(self.conn_b, return_counts=True))

        self.register_buffer('counts_ab', self.counts_a + self.counts_b)
        self.counts_a[self.counts_a == 0] = 1
        self.counts_b[self.counts_b == 0] = 1
        self.counts_ab[self.counts_ab == 0] = 1
    
    
    def select_pool(self, reset_worst=True):
        self.pool = np.random.choice(self.poolsize, self.batchsize, False)

        if reset_worst:
            worst = self.scores[self.pool].argmin()
            self.reset_values(nodes=True, edges=True, index=self.pool[worst])
            
    
    def detach_values(self):
        self.node_vals = self.node_vals.detach()
        self.edge_vals = self.edge_vals.detach()

