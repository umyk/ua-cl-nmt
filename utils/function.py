import torch
from torch.autograd.function import Function


def split_heads(input_tensor: torch.Tensor, batch_size: int, length_out: int, num_of_heads: int):
    return input_tensor.view(batch_size, length_out, num_of_heads, -1).contiguous().transpose(1, 2).contiguous()


def attention_alignment(query: torch.Tensor, key: torch.Tensor):
    return query.matmul(key.transpose(-2, -1).contiguous())


def masked_alignment(alignment: torch.Tensor, input_mask: torch.Tensor, masked_value: float):
    mask = input_mask.unsqueeze(dim=1).expand_as(alignment)
    return alignment.masked_fill(mask=mask, value=masked_value)


def softmax(input_tensor: torch.Tensor):
    probs = input_tensor.exp()
    return probs.div(probs.sum(dim=(-1, ), keepdim=True))


def logsoftmax(input_tensor: torch.Tensor):
    return softmax(input_tensor).add(1e-20).log()


def cross_entropy(predicted: torch.Tensor, true: torch.Tensor):
    return predicted.mul(true.add(1e-20).log().neg()).mean(dim=(-1, ))


def entropy(probability: torch.Tensor):
    return probability.mul(probability.add(1e-20).log().neg()).sum(dim=(-1, ))


def kl_divergence(predicted: torch.Tensor, true: torch.Tensor):
    return cross_entropy(predicted, true).sub(entropy(true))
