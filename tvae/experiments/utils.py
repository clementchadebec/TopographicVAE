import torch
import numpy as np


def make_batched_masks(data, prob_missing_data, batch_size):
    mask = torch.zeros(data.shape[:2], requires_grad=False)
    prob = ((1 - prob_missing_data) - 2 / data.shape[1]) * data.shape[1] / (data.shape[1] - 2)

    for i in range(int(data.shape[0] / batch_size)):

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        
        _mask = np.take_along_axis(_mask, idx, axis=-1)
        mask[i*batch_size:(i+1)*batch_size] = _mask.repeat(batch_size, 1)

    if data.shape[0] % batch_size > 0:

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        _mask = np.take_along_axis(_mask, idx, axis=-1)

        mask[-(data.shape[0] % batch_size):] = _mask.repeat((data.shape[0] % batch_size), 1)


    return mask