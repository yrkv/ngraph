## Neural Graph

TODO: intro

### Overview

The goal of this project is to explore the possibilities of meta-learning a learning process. For this, the project consists of two parts: tasks which are designed to require learning ability to be solved and a model designed explicitly around those tasks. 

For now, lets assume that a brain can be considered as a graph. Consider 'encoding' the relevant state of neurons and relationships between them (i.e. axons/synapses/dendrites/chemical gradients) into arbitrarily large vectors at the nodes and edges, respectively. This forms a (very large) graph which is equivalent to the brain. For a brain to think or learn, the chemical and physical processes which drive it need to happen. Notably, the physics involved is assumed to be entirely local. In terms of the graph, those processes could be considered as a function which takes a local portion of the graph and alters it in some way. Since we already know a brain can learn from experience, we can argue that there should exist some algorithm, which when applied locally on a graph containing data at nodes and edges, results in learning. 

Note that this essentially results in some variant of a graph neural network. It would be possible to use some alternate architecture, but for now a straightforward message passing neural network is sensible.

Another perspective would be to consider this as a Neural Cellular Automata with a less restricted structure, Rather than optimizing for some global pattern to emerge based on local rules, we optimize those local rules for a global behavior, similar to (that one paper).

Processing input involves feeding that input into fixed input nodes and reading outputs from fixed output nodes. The end goal is to optimize the graph's rules such that it learns from input directly through iteration of the graph, without needing gradient descent.

### Experiments

#### Language

TODO: intro

For the ngraph rules to learn a sufficiently general algorithm, we can optimize with a different language for every instance in the batch. To do this, we procedurally generate languages. By defining a 'language' as a procedurally generated context-free grammar with random rules, we can generate endless streams of text with consistent patterns for each graph. Although it's not a realistic model of natural language, it approaches the complexity with a similar modality, making it useful as a sufficiently hard toy problem. See `data.py` for the exact details on generation. Ideally, if the manifold of fake languages were broad enough to include most/all patterns of a natural language, then optimizing a system which can learn on any fake language would be sufficient for learning on that natural language.

Eventually, we will want to optimize graph rules such that a blank initialization of a graph would learn to model any given fake language just from reading it. For this, it's first necessary to demonstrate that graph rules can be optimized such that for any language within the set there is an instantiation of the graph values which models it well. We can just use gradient descent with  separately optimized graph values. The graph values are reset to the optimized ones between every batch. We can run this first experiment with the following configuraton, achieving a rather low loss.

```bash
# TODO
```

TODO: put graph here

#### Vision

TODO

#### Functions

TODO