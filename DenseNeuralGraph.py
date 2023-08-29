import torch
import torch.nn as nn
import numpy as np
import networkx as nx

class InputIntegrator(nn.Module):
    def __init__(self, ch_inp:int=8, ch_n:int=8):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ch_inp+ch_n, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n),
        )
    
    def forward(self, x):
        return self.main(x)
    
class LabelIntegrator(nn.Module):
    def __init__(self, ch_out:int=8, ch_n:int=8):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ch_out+ch_n, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n),
        )
    
    def forward(self, x):
        return self.main(x)

class OutputInterpreter(nn.Module):
    def __init__(self, ch_out:int=8, ch_n:int=8):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ch_n, 32),
            nn.SiLU(),
            nn.Linear(32, ch_out),
        )
    
    def forward(self, x):
        return self.main(x)

# Function that takes in a pair of nodes (with their info) and edge and outputs a message for each
class Message(nn.Module):
    def __init__(self, ch_n:int=8, ch_e:int=8):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(ch_n*2 + ch_e, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n*2 + ch_e),
        )
    
    def forward(self, x):
        return self.main(x)

# Takes in aggregated forward messages, backward messages, and current node state (plus info) and outputs an update for the node
class Update(nn.Module):
    def __init__(self, ch_n:int=8):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(ch_n*3, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_n),
        )

    def forward(self, x):
        return self.main(x)

# Takes in a vertex and any additional info about it and generates keys / queries
class Attention(nn.Module):
    def __init__(self, ch_n:int=8, ch_k:int=8):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(ch_n, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, ch_k*4),
        )
    
    def forward(self, x):
        return self.main(x)


class NeuralGraph(nn.Module):
    def __init__(self, n_nodes, n_inputs, n_outputs, ch_n=8, ch_e=8, ch_k=8, ch_inp=1, ch_out=1, decay=0,
                 value_range=[-100, 100], value_init="trainable", init_value_std=1, set_nodes=False, aggregation="attention", n_heads=1,
                 use_label=True, node_dropout_p=0, edge_dropout_p=0, poolsize=None, device="cpu",
                 n_models=1, message_generator=Message, update_generator=Update, attention_generator=Attention, 
                 inp_int_generator=InputIntegrator, label_int_generator=LabelIntegrator, out_int_generator=OutputInterpreter):
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
        :param value_range: The range that node and edge values can take.  Anything above 
            or below will be clamped to this range
        :param value_init: One of [trainable, trainable_batch, random, zeros].  Decides how to initialize node and edges.  
            If using trainable_batch make sure to call graph.init_vals(batch_size=___) BEFORE making an optimizer with graph.parameters().
            Otherwise the initial values will not be registered or trained.
        :param init_value_std: Standard deviation of initial values
        :param set_nodes: Instead of adding to a node's current value every timestep, it will set the node.
        :param aggregation: One of [attention, sum, mean].  How to aggregate messages
        :param device: What device to run everything on
        :param use_label: Whether to be able to input the label to an example into the output nodes of the graph for pseudolearning
        :param node_dropout_p: What percent of node updates to drop to 0
        :param edge_dropout_p: What percent of edge updates to drop to 0
        :param poolsize: Poolsize to use for persistence training (if set to None then no persistence training)

        :param n_models: Number of models to cycle through
        :param message_generator: Function to generate message models.  Must have very 
            specific shape of input_shape=(ch_n*2 + ch_e) and output_shape=(ch_n*2 + ch_e)
        :param update_generator: Function to generate update models.  Must have very 
            specific shape of input_shape=(ch_n*3) and output_shape=(ch_n)
        :param attention_generator: Function to generate attention models.  Must have very 
            specific shape of input_shape=(ch_n) and output_shape=(ch_k*4)
        :param inp_int_generator: Function to generate input integrator.  Must have very
            specific shape of input_shape=(ch_inp+ch_n) and output_shape=(ch_n)
        :param label_int_generator: Function to generate label integrator.  Must have very
            specific shape of input_shape=(ch_out+ch_n) and output_shape=(ch_n)
        :param out_int_generator: Function to generate output interpreter.  Must have very
            specific shape of input_shape=(ch_n) and output_shape=(ch_out)
        """
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.ch_n, self.ch_e, self.ch_k, self.ch_inp, self.ch_out = ch_n, ch_e, ch_k, ch_inp, ch_out
        self.decay, self.value_range, self.n_models = decay, value_range, n_models
        self.value_init, self.set_nodes, self.aggregation, self.use_label = value_init, set_nodes, aggregation, use_label
        self.init_value_std, self.n_heads = init_value_std, n_heads
        self.node_dropout_p, self.edge_dropout_p, self.poolsize = node_dropout_p, edge_dropout_p, poolsize
        self.device = device
        self.pool = None

        assert self.aggregation in ["attention", "sum", "mean"], f"Unknown aggregation option {self.aggregation}"
        assert self.ch_n % self.n_heads == 0 and self.ch_k % self.n_heads == 0, "ch_n and ch_k need to be divisible by n_heads"
        
        if self.value_init == "trainable":
            self.register_parameter("init_nodes", nn.Parameter(torch.randn(self.n_nodes, self.ch_n, device=self.device) * self.init_value_std, requires_grad=True))
            self.register_parameter("init_edges", nn.Parameter(torch.randn(self.n_nodes, self.n_nodes, self.ch_e, device=self.device) * self.init_value_std, requires_grad=True))

        self.register_buffer('nodes', torch.zeros(1, self.n_nodes, self.ch_n, device=self.device), persistent=False)
        self.register_buffer('edges', torch.zeros(1, self.n_nodes, self.n_nodes, self.ch_e, device=self.device), persistent=False)

        # If training with persistence, initialize the pool with poolsize
        if self.poolsize:
            self.init_vals(batch_size=self.poolsize)

        self.messages = nn.ModuleList([message_generator(ch_n=ch_n, ch_e=ch_e).to(self.device) for _ in range(self.n_models)])
        self.updates = nn.ModuleList([update_generator(ch_n=ch_n).to(self.device) for _ in range(self.n_models)])
        
        if self.aggregation == 'attention':
            self.multi_head_outa = nn.Linear(self.ch_n, self.ch_n, bias=False).to(self.device)
            self.multi_head_outb = nn.Linear(self.ch_n, self.ch_n, bias=False).to(self.device)
            self.attentions = nn.ModuleList([attention_generator(ch_n=ch_n, ch_k=ch_k).to(self.device) for _ in range(self.n_models)])
        
        self.inp_int = inp_int_generator(ch_inp=ch_inp, ch_n=ch_n)
        self.out_int = out_int_generator(ch_out=ch_out, ch_n=ch_n)
        if self.use_label:
            self.label_int = label_int_generator(ch_out=ch_out, ch_n=ch_n)
        
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
        m_x = torch.cat([
            torch.repeat_interleave(self.nodes[indices].unsqueeze(1), self.n_nodes, axis=1), 
            torch.repeat_interleave(self.nodes[indices].unsqueeze(2), self.n_nodes, axis=2), 
            self.edges[indices]
        ], dim=3)
        
        m = self.messages[t % self.n_models](m_x)
        m_a, m_b, m_ab = torch.split(m, [self.ch_n, self.ch_n, self.ch_e], 3)
        
        if self.aggregation == "attention":
            attention = self.attentions[t % self.n_models](self.nodes[indices])
            attention = attention.reshape(*attention.shape[:-1], self.n_heads, (self.ch_k*4)//self.n_heads)

            f_keys, f_queries, b_keys, b_queries = torch.split(attention, [self.ch_k//self.n_heads,]*4, -1)
 
            f = nn.functional.softmax(
                (torch.repeat_interleave(f_keys.unsqueeze(1), self.n_nodes, axis=1) * torch.repeat_interleave(f_queries.unsqueeze(2), self.n_nodes, axis=2)).sum(-1)
            )
            b = nn.functional.softmax(
                (torch.repeat_interleave(b_queries.unsqueeze(1), self.n_nodes, axis=1) * torch.repeat_interleave(b_keys.unsqueeze(2), self.n_nodes, axis=2)).sum(-1)
            )

            heads_b = m_b.reshape(*m_b.shape[:-1], self.n_heads, self.ch_n//self.n_heads) * torch.repeat_interleave(f.unsqueeze(-1), self.ch_n//self.n_heads, -1)
            heads_a = m_a.reshape(*m_a.shape[:-1], self.n_heads, self.ch_n//self.n_heads) * torch.repeat_interleave(b.unsqueeze(-1), self.ch_n//self.n_heads, -1)
            
            m_b = self.multi_head_outb(heads_b.reshape(*m_b.shape))
            m_a = self.multi_head_outa(heads_a.reshape(*m_a.shape))

            agg_m_a = m_a.sum(1)
            agg_m_b = m_b.sum(2)

        if self.aggregation == "sum":
            agg_m_a = m_a.sum(1)
            agg_m_b = m_b.sum(2)

        if self.aggregation == "mean":
            agg_m_a = m_a.mean(1)
            agg_m_b = m_b.mean(2)
        
        # Get updates
        u_x = torch.cat([agg_m_a, agg_m_b, self.nodes[indices]], axis=2)
        update = self.updates[t % self.n_models](u_x)

        # apply dropouts
        masked_update = update * torch.where(torch.rand_like(update) < self.node_dropout_p, 0, 1)
        masked_m_ab = m_ab * torch.where(torch.rand_like(m_ab) < self.edge_dropout_p, 0, 1)

        # Apply updates
        if nodes:
            if self.set_nodes:
                # Just set the nodes instead of adding to them
                # Ignores dt
                self.nodes[indices] = ((1-self.decay)*masked_update).clamp(*self.value_range)
            else:
                self.nodes[indices] = (self.nodes[indices] * ((1-self.decay)**dt) + masked_update*dt).clamp(*self.value_range)
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
            if self.value_init == "trainable":
                self.nodes = torch.repeat_interleave((self.init_nodes).clone().unsqueeze(0), batch_size, 0)
            elif self.value_init == "trainable_batch":
                if not hasattr(self, "init_nodes"):
                    self.register_parameter("init_nodes", nn.Parameter(torch.randn(batch_size, self.n_nodes, self.ch_n, device=self.device) * self.init_value_std, requires_grad=True))
                assert self.init_nodes.shape[0] == batch_size, "trainable_batch is on but batch size changed"
                self.nodes = self.init_nodes.clone()
            elif self.value_init == "random":
                self.nodes = torch.randn(batch_size, self.n_nodes, self.ch_n, device=self.device) * self.init_value_std
            elif self.value_init == "zeros":
                self.nodes = torch.zeros(batch_size, self.n_nodes, self.ch_n, device=self.device)
            else:
                raise RuntimeError(f"Unknown initial value config {self.value_init}")
            
        if edges:
            if self.value_init == "trainable":
                self.edges = torch.repeat_interleave((self.init_edges).clone().unsqueeze(0), batch_size, 0)
            elif self.value_init == "trainable_batch":
                if not hasattr(self, "init_edges"):
                    self.register_parameter("init_edges", nn.Parameter(torch.randn(batch_size, self.n_nodes, self.n_nodes, self.ch_e, device=self.device) * self.init_value_std, requires_grad=True))
                assert self.init_edges.shape[0] == batch_size, "trainable_batch is on but batch size changed"
                self.edges = self.init_edges.clone()
            elif self.value_init == "random":
                self.edges = torch.randn(batch_size, self.n_nodes, self.n_nodes, self.ch_e, device=self.device) * self.init_value_std
            elif self.value_init == "zeros":
                self.edges = torch.zeros(batch_size, self.n_nodes, self.n_nodes, self.ch_e, device=self.device)
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
                self.nodes[indices] = torch.randn(len(indices), self.n_nodes, self.ch_n, device=self.device) * self.init_value_std
            if edges:
                self.edges[indices] = torch.randn(len(indices), self.n_nodes, self.n_nodes, self.ch_e, device=self.device) * self.init_value_std
        elif self.value_init == "zeros":
            if nodes:
                self.nodes[indices] = torch.zeros(len(indices), self.n_nodes, self.ch_n, device=self.device)
            if edges:
                self.edges[indices] = torch.zeros(len(indices), self.n_nodes, self.n_nodes, self.ch_e, device=self.device)
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

        int_inps = self.inp_int(torch.cat([inp, self.nodes[indices, :self.n_inputs]], axis=2))
        
        if self.set_nodes:
            self.nodes[indices, :self.n_inputs] = int_inps
        else:
            self.nodes[indices, :self.n_inputs] += int_inps
        
        if label is not None:
            assert self.use_label, "Tried to apply labels but use_label was set to False"

            if not type(label) == torch.Tensor:
                label = torch.tensor(label, device=self.device)
            if self.ch_out == 1 and len(label.shape) != 3:
                label = label.unsqueeze(-1)
            

            int_label = self.label_int(torch.cat([label, self.nodes[indices, -self.n_outputs:]], axis=2))
            if self.set_nodes:
                self.nodes[indices, -self.n_outputs:] = int_label
            else:
                self.nodes[indices, -self.n_outputs:] += int_label
            
    def read_outputs(self):
        """
        Reads the outputs of the graph
        :return: outputs of shape (batch_size, n_outputs, ch_out) unless ch_out == 1 then (batch_size, n_outputs)
        """
        assert self.pool is not None or self.poolsize is None, "No pool selected but poolsize > 0"
        indices = self.pool if self.pool is not None else np.arange(len(self.nodes))

        if self.ch_out == 1:
            return self.out_int(self.nodes[indices, -self.n_outputs:]).squeeze(-1)
        return self.out_int(self.nodes[indices, -self.n_outputs:])
    
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
        
    def forward(self, x, time=5, dt=1, apply_once=False, nodes=True, edges=False):
        """
        Take input x and run the graph for a certain amount of time with a certain time resolution.
        :param x: Input to the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param apply_once: Whether to apply the input once at the beginning or every timestep
        :param nodes: Whether to update nodes
        :param edges: Whether to update edges
        """
        timesteps = round(time / dt)

        if apply_once:
            self.apply_vals(x)

        for t in range(timesteps):
            if not apply_once:
                self.apply_vals(x)
            self.timestep(nodes=nodes, edges=edges, dt=dt, t=t)
        return self.read_outputs()
    
    def backward(self, x, y, time=5, dt=1, apply_once=False, nodes=True, edges=False, edges_at_end=True):
        """
        Takes an input x and a label y and puts them in the graph and runs the graph for a certain amount of time with a certain time resolution.
        :param x: Input for the graph.  Must be shape (batch_size, n_inputs, ch_inp) unless ch_inp == 1 
            in which case it can just be (batch_size, n_inputs)
        :param y: Label for the graph.  Must be shape (batch_size, n_outputs, ch_out) unless ch_out == 1
            in which case it can just be (batch_size, n_outputs)
        :param time: time to run the graph for
        :param dt: Time resolution of the graph
        :param apply_once: Whether to apply the input once at the beginning or every timestep
        :param nodes: Whether to update nodes every timestep
        :param edges: Whether to update edges every timestep
        :param edges_at_end: Whether to update edges on the last timestep
        """

        timesteps = round(time / dt)

        if apply_once:
            self.apply_vals(x, label=y)

        for t in range(timesteps-1):
            if not apply_once:
                self.apply_vals(x, label=y)
            self.timestep(nodes=nodes, edges=edges, dt=dt, t=t)
        self.timestep(nodes=nodes, edges=edges or edges_at_end, dt=dt, t=t)
    
    def predict(self, X, time=5, dt=1, reset_nodes=False, **kwargs):
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

            preds.append(self.forward(X[:, i], dt=dt, time=time, **kwargs))
        return torch.stack(preds, axis=1)
    
    def learn(self, X, Y, time=5, dt=1, reset_nodes=False, **kwargs):
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

            self.forward(X[:, i], dt=dt, time=time, **kwargs)
            self.backward(X[:, i], Y[:, i], dt=dt, time=time, **kwargs)
    
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
