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


def test_gather():
    t = torch.tensor([[1, 2],
                      [3, 4]])

    print(torch.gather(t, 0, torch.LongTensor([[0, 1]])))


def test_001():
    a = torch.Tensor([[1, 2],
                      [3, 4]])

    b = torch.gather(a, dim=-1, index=torch.LongTensor([[1], [1]]))

    print('a = ', a)
    print('b = ', b)


def test_gpa_1():
    point = [2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    score = [88, 93, 85, 85, 74, 75, 81, 80, 84, 81, 87, 93, 88, 91]
    score_point = 0
    for p, s in zip(point, score):
        score_point += p * s
    weight_score = score_point / sum(point)
    gpa = weight_score / 100 * 4
    print(gpa)


def test_gpa_2():
    lianghao = 87
    point = [3, 2, 4, 3, 3, 3, 1, 5, 3,
             4, 3, 3, 1, 5, 1, 1, 6, 1, 5,
             2, 5, 2, 4, 4, 1, 2, 1, 2,
             3, 1, 2, 2, 1, 2, 4, 3, 2.5, 4, 1, 2.5,
             2, 3, 2, 3, 2.5, 2, 0.5, 2, 4, 3, 3,
             3, 3, 2, 2.5, 4, 3, 1.5, 1, 2, 2, 1]
    score = [68, 92, 75, 77, lianghao, 83, 91, 90, 78,
             81, 89, 87, 87, 89, lianghao, 73, 76, 77, 93,
             90, 62.5, 95, 85, 81, lianghao, 90, 79, 80,
             85, 62, 83, 62, 82, 90, lianghao, 79, 80, 70, 75, 93,
             lianghao, 74, 61, 88, 67, 85, 84, 82, 85, 78, 73,
             81, 77, lianghao, 88, 80, 89, 84, lianghao, lianghao, 87, 86]
    score_point = 0
    for p, s in zip(point, score):
        score_point += p * s
    weight_score = score_point / sum(point)
    gpa = weight_score / 100 * 4
    print(gpa)


if __name__ == '__main__':
    # test_001()
    test_gpa_2()
