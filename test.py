import torch
from torch.distributions import Categorical
from torch import nn
import numpy as np
from datetime import datetime

def test01():
    probs = torch.FloatTensor([0.05, 0.55, 0.85])
    # probs = torch.FloatTensor([0.05, 0.05, 0.9])
    dist = Categorical(probs)
    print(dist)
    print(dist.probs)
    print(dist.log_prob(torch.as_tensor(1, dtype=torch.int32)))
    index = dist.sample()
    print(index)
    print(index.numpy())
    # print(index.item())


def test02():
    prob = torch.tensor([0.1, 0.2, 0.4, 0.3])
    new_prob = torch.tensor([0.2, 0.1, 0.3, 0.4])
    logprob = torch.log(prob)
    # print(logprob)
    new_logprob = torch.log(new_prob)
    # print(new_logprob)
    ratio = (new_logprob - logprob).exp()

    # ratio2  = new_logprob/prob
    # ratio2 = torch.div(new_logprob, logprob)
    # print(ratio)
    # print(ratio2)
    adv_v = torch.tensor([-1])
    # adv_v = torch.tensor([-1, -1, -1, -1])
    surrogate1 = adv_v * ratio
    print(surrogate1)
    ratio_clip = 0.2
    surrogate2 = adv_v * ratio.clamp(1 - ratio_clip, 1 + ratio_clip)
    print(surrogate2)
    ob = torch.min(surrogate1, surrogate2)
    print(ob)
    obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
    # print(obj_surrogate)


def test03():
    m = nn.Softmax(dim=1)
    input = torch.tensor([[0.5, 0.5, 1], [0.5, 0.5, 1]])
    output = m(input)
    print(output)


def test_dim():
    y = torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    ])

    print(torch.sum(y, dim=0))
    print(torch.sum(y, dim=1))
    print(torch.sum(y, dim=2))


def test_index():
    arr = [1, 2, 3, 4, 5]
    arr.reverse()
    print(arr)


def test_arange():
    batch_start = np.arange(0, 10, 3)
    indices = np.arange(10, dtype=np.int64)
    batches = [indices[i:i + 3] for i in batch_start]


def test_bool():
    print(1 - False)
    print(1 - True)


def test_unsq():
    state = [0.01, 0.5, 0.9]
    state = torch.tensor(state, dtype=torch.float32)
    state = state.unsqueeze(0)
    print(state)

def test_time():
    data_str = datetime.now()

    print(data_str)

if __name__ == '__main__':
    test_time()
