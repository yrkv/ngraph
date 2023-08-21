import torch
import torch.nn as nn
import numpy as np
import networkx as nx

# Function that takes in a pair of nodes (with their info) and edge and outputs a message for each
class Message(nn.Module):
    def __init__(self, ch_n:int=8, ch_e:int=8, ch_extra:int=6):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear((ch_n+ch_extra)*2 + ch_e, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n*2 + ch_e),
        )
    
    def forward(self, x):
        return self.main(x)

# Takes in aggregated forward messages, backward messages, and current node state (plus info) and outputs an update for the node
class Update(nn.Module):
    def __init__(self, ch_n:int=8, ch_extra:int=6):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(ch_n*3 + ch_extra, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n),
        )

    def forward(self, x):
        return self.main(x)

# Takes in a vertex and any additional info about it and generates keys / queries
class Attention(nn.Module):
    def __init__(self, ch_n:int=8, ch_k:int=8, ch_extra:int=6):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(ch_n+ch_extra, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_k*4),
        )
    
    def forward(self, x):
        return self.main(x)


class NeuralGraph(nn.Module):
    def __init__(self, n_nodes, n_inputs, n_outputs, connections, ch_n=8, ch_e=8, ch_k=8, ch_inp=1, ch_out=1, decay=0, leakage=0,
                 value_range=[-100, 100], value_init="trainable", set_nodes=False, aggregation="attention", device="cpu", 
                 node_dropout_p=0, edge_dropout_p=0, poolsize=None,
                 n_models=1, message_generator=Message, update_generator=Update, attention_generator=Attention):
        """
        Creates a Neural Graph.  A Neural Graph is an arbitrary directed graph which has input nodes and output nodes.  
        Every node and edge in the graph has a state which is represented as a vector with dimensionality of ch_n and ch_e repectively.
        Information flows through the graph according to a set of rules determined by 2-3 functions.
        Each timestep every triplet of (node_a, node_b, edge_ab) calculates three "messages" (m_a, m_b, m_ab) according to a message function.
        The edge message m_ab can be simply added to the edge value, however each node might have many messages from numerous connections.
        Therefore, these messages are aggregated using sum (or an attention function) and then passed to the update function, which will
        take the aggregated messages and produce the node update which is added to the node values.  This is the end of one timestep.


        :param n_nodes: Total number of nodes in the graph
        :param n_inputs: Number of input nodes in the graph (indices 0:n_inputs)
        :param n_outputs: Number of output nodes in the graph (indices -n_ouputs:-1)
        :param connections: List of connections of nodes e.g. [(0, 1), (1, 2), ... (0, 2)]
        
        :param ch_n: Number of channels for each node
        :param ch_e: Number of channels for each edge
        :param ch_k: Number of channels for attention keys/queries
        :param ch_inp: Number of input channels
        :param ch_out: Number of output channels
        
        :param decay: Amount node values are decayed each unit of time
        :param leakage: Given a node A, the average values of all nodes connected to A 
            (in either direction) will be averaged and put into A at each timestep.
            It will be combined with A's original value according to the equation
            nodeA = (1-leakage) * nodeA + leakage * avg_connect_node
        :param value_range: The range that node and edge values can take.  Anything above 
            or below will be clamped to this range
        :param value_init: One of [trainable, trainable_batch, random, zeros].  Decides how to initialize node and edges
        :param set_nodes: Instead of adding to a node's current value every timestep, it will set the node.
        :param aggregation: One of [attention, sum, mean].  How to aggregate messages
        :param device: What device to run everything on
        :param node_dropout_p: What percent of node updates to drop to 0
        :param edge_dropout_p: What percent of edge updates to drop to 0
        :param poolsize: Poolsize to use for persistence training (if set to None then no persistence training)

        :param n_models: Number of models to cycle through
        :param message_generator: Function to generate message models.  Must take as input ch_n, ch_e and ch_extra and have very 
            specific shape of input_shape=((ch_n + ch_extra)*2 + ch_e) and output_shape=(ch_n*2 + ch_e)
        :param update_generator: Function to generate update models.  Must take as input ch_n and ch_extra and have very 
            specific shape of input_shape=(ch_n*3 + ch_extra) and output_shape=(ch_n)
        :param attention_generator: Function to generate attention models.  Must take as input ch_n, ch_k and ch_extra and have very 
            specific shape of input_shape=(ch_n + ch_extra) and output_shape=(ch_k*4)
        """
        super().__init__()
        
        assert ch_n >= ch_out, f"Number of output channels ({ch_out}) is greater than total node channels ({ch_n})"
        
        self.n_nodes, self.n_edges, self.connections = n_nodes, len(connections), connections
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.ch_n, self.ch_e, self.ch_k, self.ch_inp, self.ch_out = ch_n, ch_e, ch_k, ch_inp, ch_out
        self.decay, self.value_range, self.leakage, self.n_models = decay, value_range, leakage, n_models
        self.value_init, self.set_nodes, self.aggregation = value_init, set_nodes, aggregation
        self.node_dropout_p, self.edge_dropout_p, self.poolsize = node_dropout_p, edge_dropout_p, poolsize
        self.device = device
        self.pool = None

        assert self.aggregation in ["attention", "sum", "mean"], f"Unknown aggregation option {self.aggregation}"

        self.ch_extra = self.ch_inp + self.ch_out + 4
        # extra channels are
        # 0:ch_inp is inp
        # ch_inp:ch_inp+ch_out is label

        # -4:hasLabel
        # -3:isInp
        # -2:isHid
        # -1:isOut
        
        if self.value_init == "trainable":
            self.register_parameter("init_nodes", nn.Parameter(torch.randn(self.n_nodes, self.ch_n, device=self.device), requires_grad=True))
            self.register_parameter("init_edges", nn.Parameter(torch.randn(self.n_edges, self.ch_e, device=self.device), requires_grad=True))

        self.register_buffer('nodes', torch.zeros(1, self.n_nodes, self.ch_n, device=self.device), persistent=False)
        self.register_buffer('node_info', torch.zeros(1, self.n_nodes, self.ch_extra, device=self.device), persistent=False)
        self.register_buffer('edges', torch.zeros(1, self.n_edges, self.ch_e, device=self.device), persistent=False)

        # If training with persistence, initialize the pool with poolsize
        if self.poolsize:
            self.init_vals(batch_size=self.poolsize)

        self.messages = nn.ModuleList([message_generator(ch_n=ch_n, ch_e=ch_e, ch_extra=self.ch_extra).to(self.device) for _ in range(self.n_models)])
        self.updates = nn.ModuleList([update_generator(ch_n=ch_n, ch_extra=self.ch_extra).to(self.device) for _ in range(self.n_models)])
        
        if self.use_attention:
            self.attentions = nn.ModuleList([attention_generator(ch_n=ch_n, ch_k=ch_k, ch_extra=self.ch_extra).to(self.device) for _ in range(self.n_models)])
        
        conn_a, conn_b = zip(*connections)
        self.conn_a = torch.tensor(conn_a).long().to(self.device)
        self.conn_b = torch.tensor(conn_b).long().to(self.device)
        
        self.counts_a = torch.zeros(self.n_nodes, device=self.device).long()
        self.counts_b = torch.zeros(self.n_nodes, device=self.device).long()
        self.counts_a.index_add_(0, *torch.unique(self.conn_a, return_counts=True))
        self.counts_b.index_add_(0, *torch.unique(self.conn_b, return_counts=True))
        self.counts_a[self.counts_a == 0] = 1
        self.counts_b[self.counts_b == 0] = 1
        
    def timestep(self, dt=1, nodes=True, edges=True, t=0):
        """
        Runs one timestep of the Neural Graph.
        :param dt: The temporal resolution of the timestep.  At the limit of dt->0, the rules become differential equations.
            This was inspired by smoothlife/lenia
        :param nodes: Whether to update node values this timestep
        :param edges: Whether to update edge values this timestep
        :param t: The current timestep (used to decide which set of models to use)
        """

        # t is what cycles through the models

        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize > 0"

        indices = self.pool if self.pool is not None else np.arange(len(self.nodes))

        # Get messages
        node_data = torch.cat([self.node_info[indices], self.nodes[indices]], axis=2)
        m_x = torch.cat([node_data[:, self.conn_a], node_data[:, self.conn_b], self.edges[indices]], dim=2)
        m = self.messages[t % self.n_models](m_x)
        
        m_a, m_b, m_ab = torch.split(m, [self.ch_n, self.ch_n, self.ch_e], 2)
        
        if self.aggregation == "attention":
            attention = self.attentions[t % self.n_models](node_data)
            f_keys, f_queries, b_keys, b_queries = torch.split(attention, [self.ch_k,]*4, -1)
            
            f_attention = torch.softmax((f_keys[:, self.conn_a] * f_queries[:, self.conn_b]).sum(-1), -1).unsqueeze(-1)
            b_attention = torch.softmax((b_queries[:, self.conn_a] * b_keys[:, self.conn_b]).sum(-1), -1).unsqueeze(-1)
            
            m_b = m_b * f_attention
            m_a = m_a * b_attention

        
        # Aggregate messages (summing up for now.  Could make it average instead)
        agg_m_a = torch.zeros(len(indices), self.n_nodes, self.ch_n, device=self.device)
        agg_m_b = torch.zeros(len(indices), self.n_nodes, self.ch_n, device=self.device)
        agg_m_a.index_add_(1, self.conn_a, m_a)
        agg_m_b.index_add_(1, self.conn_b, m_b)

        if self.aggregation == "mean":
            agg_m_a.divide_(self.counts_a[None, :, None])
            agg_m_b.divide_(self.counts_b[None, :, None])
        
        # Get updates
        u_x = torch.cat([agg_m_a, agg_m_b, node_data], axis=2)
        update = self.updates[t % self.n_models](u_x)

        # apply dropouts
        masked_update = update * torch.where(torch.rand_like(update) < self.node_dropout_p, 0, 1)
        masked_m_ab = m_ab * torch.where(torch.rand_like(m_a) < self.edge_dropout_p, 0, 1)

        # Calculate leakage for each node
        agg_leakage = torch.zeros(len(indices), self.n_nodes, self.ch_n, device=self.device)
        agg_leakage.index_add_(1, self.conn_a, self.nodes[indices][:, self.conn_b])
        agg_leakage.index_add_(1, self.conn_b, self.nodes[indices][:, self.conn_a])
        agg_leakage.divide_(torch.repeat_interleave((self.counts_a+self.counts_b).unsqueeze(1), self.ch_n, 1))


        # Apply updates
        if nodes:
            if self.set_nodes:
                # Just set the nodes instead of adding to them
                # Ignores dt
                self.nodes[indices] = ((1-self.leakage)*(1-self.decay)*masked_update+self.leakage*agg_leakage).clamp(*self.value_range)
            else:
                # Node values get decayed and leaked into

                # Decay node state and add update
                new_node_state = self.nodes[indices] * ((1-self.decay)**dt) + masked_update*dt
                # Leak in other connected node states and clamp
                self.nodes[indices] = ((1-self.leakage)*new_node_state+self.leakage*agg_leakage).clamp(*self.value_range)

        if edges:
            self.edges[indices] = (self.edges[indices] + masked_m_ab).clamp(*self.value_range)
        
    def init_vals(self, nodes=True, edges=True, batch_size=1):
        """
        Initialize nodes and edges.
        :param nodes: Whether to initialize nodes
        :param edges: Whether to initialize edeges
        :batch_size: Batch size of the initialization
        """
        if nodes:
            # Figure out dynamically with batch_size
            self.node_info = torch.zeros(batch_size, self.n_nodes, self.ch_extra, device=self.device)
            self.node_info[:, :self.n_inputs, -3] = 1
            self.node_info[:, self.n_inputs:-self.n_outputs, -2] = 1
            self.node_info[:, -self.n_outputs:, -1] = 1
            
            if self.value_init == "trainable":
                self.nodes = torch.repeat_interleave((self.init_nodes).clone().unsqueeze(0), batch_size, 0)
            elif self.value_init == "trainable_batch":
                if not hasattr(self, "init_nodes"):
                    self.register_parameter("init_nodes", nn.Parameter(torch.randn(batch_size, self.n_nodes, self.ch_n, device=self.device), requires_grad=True))
                assert self.init_nodes.shape[0] == batch_size, "trainable_batch is on but batch size changed"
                self.nodes = self.init_nodes.clone()
            elif self.value_init == "random":
                self.nodes = torch.randn(batch_size, self.n_nodes, self.ch_n, device=self.device) * .01
            elif self.value_init == "zeros":
                self.nodes = torch.zeros(batch_size, self.n_nodes, self.ch_n, device=self.device)
            else:
                raise RuntimeError(f"Unknown initial value config {self.value_init}")
            
        if edges:
            if self.value_init == "trainable":
                self.edges = torch.repeat_interleave((self.init_edges).clone().unsqueeze(0), batch_size, 0)
            elif self.value_init == "trainable_batch":
                if not hasattr(self, "init_edges"):
                    self.register_parameter("init_edges", nn.Parameter(torch.randn(batch_size, self.n_edges, self.ch_e, device=self.device), requires_grad=True))
                assert self.init_edges.shape[0] == batch_size, "trainable_batch is on but batch size changed"
                self.edges = self.init_edges.clone()
            elif self.value_init == "random":
                self.edges = torch.randn(batch_size, self.n_edges, self.ch_e, device=self.device) * .01
            elif self.value_init == "zeros":
                self.edges = torch.zeros(batch_size, self.n_edges, self.ch_e, device=self.device)
            else:
                raise RuntimeError(f"Unknown initial value config {self.value_init}")
    
    # Reset just one index
    def reset_vals(self, indices=None, nodes=True, edges=True):
        indices = indices or np.arange(len(self.nodes))
        """
        Reset certain values in pool (or whole batch)
        :param indices: indices in the pool (or batch) to reset
        :param nodes: Whether to reset nodes
        :param edges: Whether to reset edges
        """
        if self.value_init == "trainable":
            if nodes:
                self.nodes[indices] = torch.repeat_interleave(self.init_nodes.clone().unsqueeze(0), len(indices), 0)
            if edges:
                self.edges[indices] = torch.repeat_interleave(self.init_edges.clone().unsqueeze(0), len(indices), 0)
        elif self.value_init == "trainable_batch":
            if nodes:
                self.nodes[indices] = self.init_nodes[indices].clone()
            if edges:
                self.edges[indices] = self.init_edges[indices].clone()  
        elif self.value_init == "random":
            if nodes:
                self.nodes[indices] = torch.randn(len(indices), self.n_nodes, self.ch_n, device=self.device) * .01
            if edges:
                self.edges[indices] = torch.randn(len(indices), self.n_edges, self.ch_e, device=self.device) * .01
        elif self.value_init == "zeros":
            if nodes:
                self.nodes[indices] = torch.zeros(len(indices), self.n_nodes, self.ch_n, device=self.device)
            if edges:
                self.edges[indices] = torch.zeros(len(indices), self.n_edges, self.ch_e, device=self.device)
        else:
            raise RuntimeError(f"Unknown initial value config {self.value_init}")

    def select_pool(self, batch_size=1, explosion_threshold=None, reset=True, losses=None):
        """
        Set the current batch to a random set of the pool of values

        :param batch_size: The batch_size of the batch
        :param explosion_threshold: If the average value of a graph is above this then reset it
        :param reset: Whether to reset a graph in the pool (if no losses are provided it will be a random one)
        :param losses: The losses of each graph in the batch (the graph with the highest loss will be the one that's reset)
        """

        assert self.poolsize is not None, "Poolsize was None"
        
        self.pool = np.random.choice(self.poolsize, batch_size, False)

        # Reset any exploded graphs
        if explosion_threshold:
            for i in self.pool:
                if (self.nodes[i].abs().mean() + self.edges[i].abs().mean()) / 2 > explosion_threshold:
                    self.reset_vals(indices=np.array([i]))

        # Reset either one random graph or the graph with the highest loss if provided with losses
        if reset:
            if losses is not None:
                i = self.pool[np.argmax(losses)]
            else:
                i = np.random.randint(self.poolsize)
            self.reset_vals(indices=([i]))

    def apply_vals(self, inp, label=None):
        """
        Apply inputs to input nodes in the graph and labels to output nodes in the graph
        :param inp: The inputs for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param label: The labels for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        """

        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize was not None"
        indices = self.pool if self.pool is not None else np.arange(len(self.nodes))

        if not type(inp) == torch.Tensor:
            inp = torch.tensor(inp, device=self.device)
        if self.ch_inp == 1 and len(inp.shape) != 3:
            inp = inp.unsqueeze(-1)
        
        # Clear out old input and labels and hasLabel
        self.node_info[indices, :, :self.ch_inp+self.ch_out+1] = 0
        self.node_info[indices, :self.n_inputs, :self.ch_inp] = inp.clone()
        
        if label is not None:
            if not type(label) == torch.Tensor:
                label = torch.tensor(label, device=self.device)
            if self.ch_out == 1 and len(label.shape) != 3:
                label = label.unsqueeze(-1)
            
            # hasLabel = 1
            self.node_info[indices, -self.n_outputs:, self.ch_inp:self.ch_inp+self.ch_out] = label.clone()
            self.node_info[indices, :, -4] = 1
            
    def read_outputs(self):
        """
        Reads the outputs of the graph
        :return: outputs of shape (batch_size, n_outputs, ch_out) unless ch_out == 1 then (batch_size, n_outputs)
        """
        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize > 0"
        indices = self.pool if self.pool is not None else np.arange(len(self.nodes))

        if self.ch_out == 1:
            return self.nodes[indices, -self.n_outputs:, 0].clone()
        return self.nodes[indices, -self.n_outputs:, :self.ch_out].clone()
    
    def overflow(self, k=5):
        """
        Calculates the overflow of the graph intended to be added to loss to prevent explosions.
        :param k: maximum value of node and edge values before they incur a penalty.
        """

        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize > 0"
        indices = self.pool if self.pool is not None else np.arange(len(self.nodes))

        node_overflow = ((self.nodes[indices] - self.nodes[indices].clamp(-k, k))**2).mean()
        edge_overflow = ((self.edges[indices] - self.edges[indices].clamp(-k, k))**2).mean()
        return node_overflow + edge_overflow
        
    def forward(self, x, time=5, dt=1, nodes=True, edges=False):
        """
        Take input x and run the graph for a certain amount of time with a certain time resolution.
        :param x: Input to the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param nodes: Whether to update nodes
        :param edges: Whether to update edges
        """
        timesteps = round(time / dt)
        self.apply_vals(x)

        for t in range(timesteps):
            self.timestep(nodes=nodes, edges=edges, dt=dt, t=t)
        return self.read_outputs()
    
    def backward(self, x, y, time=5, dt=1, nodes=True, edges=False, edges_at_end=True):
        """
        Takes an input x and a label y and puts them in the graph and runs the graph for a certain amount of time with a certain time resolution.
        :param x: Input for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param y: Label for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param nodes: Whether to update nodes every timestep
        :param edges: Whether to update edges every timestep
        :param edges_at_end: Whether to update edges on the last timestep
        """

        timesteps = round(time / dt)
        self.apply_vals(x, label=y)

        for t in range(timesteps-1):
            self.timestep(nodes=nodes, edges=edges, dt=dt, t=t)
        self.timestep(nodes=nodes, edges=edges or edges_at_end, dt=dt, t=t)
    
    def predict(self, X, time=5, dt=1, reset_nodes=False):
        """
        Take a series of inputs and one by one inserts them into the graph and runs a forward pass.
        :param X:  Inputs for the graph.  Must be shape (batch_size, n_examples, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_examples, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :reset_nodes: Whether to reset the nodes after every forward pass

        :return: The outputs of the graph after the examples with shape (batch_size, n_examples, n_outputs, ch_out) unless ch_out == 1 
            in which case it is (batch_size, n_examples, n_outputs)
        """
        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize was not None"

        preds = []
        for i in range(X.shape[1]):
            if reset_nodes:
                self.reset_vals(indices=self.pool, nodes=True, edges=False)

            preds.append(self.forward(X[:, i], dt=dt, time=time))
        return torch.stack(preds, axis=1)
    
    def learn(self, X, Y, time=5, dt=1, reset_nodes=False):
        """
        Take a series of inputs and inserts them into the graph, runs a forward pass and then insert the correspond labels into the graph
        and runs a backwards pass.  
        :param X:  Inputs for the graph.  Must be shape (batch_size, n_examples, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_examples, n_inputs)
        :param Y:  Labels for the graph.  Must be shape (batch_size, n_examples, n_outputs, ch_out) unless ch_out == 1 
            in which case it can just be (batch_size, n_examples, n_outputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :reset_nodes: Whether to reset the nodes after every forward/backward pair
        """
        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize was not None"

        for i in range(X.shape[1]):
            if reset_nodes:
                self.reset_vals(indices=self.pool, nodes=True, edges=False)

            self.forward(X[:, i], dt=dt, time=time)
            self.backward(X[:, i], Y[:, i], dt=dt, time=time)
    
    def detach_vals(self):
        """
        Detach the node and edge values from the torch compute graph.
        """
        self.nodes = self.nodes.detach()
        self.edges = self.edges.detach()

    def save_rules(self, path):
        for name, model in {"_m":self.messages, "_u":self.updates, "_a":self.attentions}.items():
            torch.save(model.state_dict(), f"{path}{name}.pth")

    def load_rules(self, path):
        for name, model in {"_m":self.messages, "_u":self.updates, "_a":self.attentions}.items():
            model.load_state_dict(torch.load(f"{path}{name}.pth"))

    def save(self, path):
        self.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def plot(self):
        g = nx.DiGraph()
        g.add_nodes_from(list(range(self.n_nodes)))
        g.add_edges_from(self.connections)

        nx.draw(g)