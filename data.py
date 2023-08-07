import numpy as np
from collections import namedtuple


# utility functions
def rolling_window(xs: np.ndarray, size: int):
    shape = xs.shape[:-1] + (xs.shape[-1] - size + 1, size)
    strides = xs.strides + (xs.strides[-1],)
    return np.lib.stride_tricks.as_strided(xs, shape=shape, strides=strides)

def find_subarrays(xs: np.ndarray, ys: np.ndarray):
    if len(ys) > len(xs):
        return np.array([], dtype='int64')
    mask = (rolling_window(xs, len(ys)) == ys).all(axis=1)
    return np.mgrid[:len(mask)][mask]


def define_expansion(a, b):
    expand_map = []
    for i in range(0, a):
        targets = []
        for _ in range(np.random.geometric(0.65)):
            k = min(np.random.geometric(0.75) + 1, int(np.floor(b**0.5)))
            targets.append((np.random.choice(b, k), np.random.rand()))

        mapping, weights = zip(*targets)
        weights = np.array(weights) / np.sum(weights)
        expand_map.append((mapping, weights))
    return(expand_map)


# function: tokens -> new_tokens
# Rule = namedtuple('Rule', ['function', 'uniqueness'])

# k: number of tokens
def define_rule(k):
    return np.random.choice([
        define_rule_a,
        define_rule_b,
    ])(k)

# fixed replacement of a sequence
def define_rule_a(k):
    # k_a = min(np.random.geometric(0.75) + 1, int(np.floor(k**0.5)))
    # k_b = min(np.random.geometric(0.75) + 1, int(np.floor(k**0.5)))
    k_a = np.random.geometric(0.75) + 1
    k_b = np.random.geometric(0.75) + 1

    pat = np.random.choice(k, k_a)
    replacement = np.random.choice(k, k_b)

    # print(f'{pat=} {replacement=}')

    def rule(tokens: np.ndarray):
        indices = []
        for i in find_subarrays(tokens, pat):
            if len(indices) == 0 or i >= indices[-1] + len(pat):
                indices.append(i)
        
        if len(indices) == 0:
            return tokens
        
        to_concat = [tokens[:indices[0]], replacement]
        for a, b in zip(indices[:-1], indices[1:]):
            to_concat.extend([tokens[a+len(pat):b], replacement])
        to_concat.append(tokens[indices[-1]+len(pat):])

        return np.concatenate(to_concat)
    
    return rule, (1, tuple(pat))


# token enforces lexicographic order near it
def define_rule_b(k):
    token = np.random.choice(k)
    len_a = np.random.geometric(0.75)
    len_b = np.random.geometric(0.75)

    # 0: both before, 1: before/after, 2: both after
    mode = np.random.choice(3)
    a = [ -len_b - len_a,       1,              -len_a  ][mode]
    b = [ -len_b,               1 + len_a,      1       ][mode]

    # print(f'{a=} {len_a=} {b=} {len_b=} {token=}')

    def rule(tokens: np.ndarray):
        indices = []
        for i in find_subarrays(tokens, np.array([token])):
            index_a, index_b = i + a, i + b
            if not (0 <= index_a < len(tokens) and 0 < index_b+len_b <= len(tokens)):
                continue
            region_a = tokens[index_a:index_a+len_a]
            region_b = tokens[index_b:index_b+len_b]
            if sum(region_a == token) > 0 or sum(region_b == token) == 0:
                indices.append(i)
        
        out = tokens.copy()
        for i in indices:
            index_a, index_b = i + a, i + b
            region_a = tokens[index_a:index_a+len_a]
            region_b = tokens[index_b:index_b+len_b]
            # print(f'{region_a=} {region_b=}')
            if tuple(region_b) < tuple(region_a):
                out[index_a:index_b+len_b] = np.concatenate([
                    region_b, [token] if a < 0 and b > 0 else [], region_a
                ])

        return out
    
    return rule, (2, token)

"""
The 'language' is defined in terms of stages of expanding tokens and rules
at each stage. At each stage before the end, the tokens first get expanded
according to the expansion map. Then, rules are applied to replace or shift
tokens in consistent ways.

Ideally, this should result in an infinitely generatable sequence of tokens
with many complex interactions. The purpose of this is to act as a (toy)
dataset for language modeling.

Some example rules;
[x] "ABC -> XY" : fixed consistent replacement for group of tokens
[x] "XAY -> "YAX" : token enforces ordering near it
[ ] "A * B * C -> A * A * A" : first instance of a set of tokens replaces rest
    - this would be a long range dependency, which may be difficult to model
[ ] "A[XYZ] -> "AA[XYZ]" : token doubles under condition
[ ] "__B__ -> __B__A" : token disappears under condition
"""
class Language:
    
    def __init__(self, stages=[10, 10, 20]):
        self.stages = stages

        self.expansions = []
        self.rules = []
        for a, b in zip(stages[:-1], stages[1:]):
            self.expansions.append(define_expansion(a, b))
            stage_rules = {}
            for _ in range(b):
                rule, uniqueness = define_rule(b)
                stage_rules[uniqueness] = rule
            self.rules.append(stage_rules.values())

    def generate(self, baselen=20):
        tokens = np.random.choice(self.stages[0], baselen)

        for stage in range(len(self.expansions)):
            tokens = self.apply_expand(tokens, stage)
            tokens = self.apply_rules(tokens, stage)
        
        return tokens

    def apply_expand(self, tokens, stage) -> np.ndarray:
        out = []
        for t in tokens:
            mappings, weights = self.expansions[stage][t]
            i = np.random.choice(len(mappings), p=weights)
            out.extend(mappings[i])
        return np.array(out)
 
    def apply_rules(self, tokens, stage):
        for rule in self.rules[stage]:
            tokens = rule(tokens)
        return tokens
