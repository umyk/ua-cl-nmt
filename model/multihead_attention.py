import torch
from torch import nn
from torch.nn.functional import softmax

from model.layer_norm import Identity, LayerNorm, UnlearnableLayerNorm


class MultiHeadAttention1(nn.Module):
    def __init__(self,
                 emb_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(MultiHeadAttention1, self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_of_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.masked_value = float('-inf')

        self.attention_norm_factor = (self.emb_size / self.num_of_heads) ** 0.5

        self.linear_in_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size * 3)), requires_grad=True)
        self.linear_in_bias = nn.Parameter(torch.zeros(size=(self.emb_size * 3, )), requires_grad=True)

        self.linear_out_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_out_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        self.dropout_attention = nn.Dropout(p=self.attention_dropout_prob)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_qkv_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_pre == 'static':
            self.layer_norm_qkv_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_qkv_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_v_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_v_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_v_post = Identity()

        return

    def init_parameters(self):
        bound = (1.5 / self.emb_size) ** 0.5
        self.linear_in_weight.uniform_(-bound, bound)
        bound = (1 / self.emb_size) ** 0.5
        self.linear_in_bias.uniform_(-bound, bound)
        bound = (3 / self.emb_size) ** 0.5
        self.linear_out_weight.uniform_(-bound, bound)
        self.linear_out_bias.uniform_(-bound, bound)

        return

    def forward(self, input_qkv, input_mask):
        normalized_qkv = self.layer_norm_qkv_pre(input_qkv)
        batch_size, length_out, _ = input_qkv.size()
        query, key, value = torch.split(torch.add(torch.matmul(normalized_qkv, self.linear_in_weight),
                                                  self.linear_in_bias), self.emb_size, dim=-1)

        query = query.view(batch_size, length_out, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()
        key = key.view(batch_size, length_out, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()
        value = value.view(batch_size, length_out, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()

        alignments = query.matmul(key.transpose(-2, -1).contiguous()) / self.attention_norm_factor
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))

        context_vector = alignment_scores.matmul(value).transpose(1, 2).contiguous(). \
            view(batch_size, length_out, -1).contiguous()
        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight),
                                                 self.linear_out_bias))
        return self.layer_norm_v_post(output + input_qkv)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class MultiHeadAttention2(nn.Module):
    def __init__(self,
                 emb_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(MultiHeadAttention2, self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_of_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.masked_value = float('-inf')

        self.attention_norm_factor = (self.emb_size / self.num_of_heads) ** 0.5

        self.linear_q_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_q_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)
        self.linear_kv_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size * 2)), requires_grad=True)
        self.linear_kv_bias = nn.Parameter(torch.zeros(size=(self.emb_size * 2, )), requires_grad=True)
        self.linear_out_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_out_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        self.dropout_attention = nn.Dropout(p=self.attention_dropout_prob)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_q_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            # self.layer_norm_q_pre = Identity()
            # self.layer_norm_kv_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_kv_pre = Identity()
        elif self.layer_norm_pre == 'static':
            self.layer_norm_q_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            # self.layer_norm_q_pre = Identity()
            # self.layer_norm_kv_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_kv_pre = Identity()
        else:
            self.layer_norm_q_pre = Identity()
            self.layer_norm_kv_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_v_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_v_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_v_post = Identity()

        return

    def init_parameters(self):
        bound = (2 / self.emb_size) ** 0.5
        self.linear_kv_weight.uniform_(-bound, bound)
        bound = (1.5 / self.emb_size) ** 0.5
        self.linear_kv_bias.uniform_(-bound, bound)
        bound = (3 / self.emb_size) ** 0.5
        self.linear_q_weight.uniform_(-bound, bound)
        self.linear_q_bias.uniform_(-bound, bound)
        self.linear_out_weight.uniform_(-bound, bound)
        self.linear_out_bias.uniform_(-bound, bound)

        return

    def forward(self, input_q, input_kv, input_mask):
        normalized_q = self.layer_norm_q_pre(input_q)
        normalized_kv = self.layer_norm_kv_pre(input_kv)
        length_in = input_kv.size(1)
        batch_size, length_out, _ = input_q.size()

        query = torch.add(torch.matmul(normalized_q, self.linear_q_weight), self.linear_q_bias)
        key, value = torch.split(torch.add(torch.matmul(normalized_kv, self.linear_kv_weight), self.linear_kv_bias),
                                 self.emb_size, dim=-1)

        query = query.view(batch_size, length_out, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()
        key = key.view(batch_size, length_in, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()
        value = value.view(batch_size, length_in, self.num_of_heads, -1).contiguous().transpose(1, 2).contiguous()

        alignments = query.matmul(key.transpose(-2, -1).contiguous()) / self.attention_norm_factor
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))

        context_vector = alignment_scores.matmul(value).transpose(1, 2).contiguous(). \
            view(batch_size, length_out, -1).contiguous()
        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight),
                                                 self.linear_out_bias))
        return self.layer_norm_v_post(output + input_q)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(MultiHeadAttention, self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_of_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.masked_value = float('-inf')

        self.attention_norm_factor = (self.emb_size / self.num_of_heads) ** 0.5

        self.linear_q = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_k = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_v = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_out = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)

        self.dropout_attention = nn.Dropout(p=self.attention_dropout_prob)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_q_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_k_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_v_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_pre == 'static':
            self.layer_norm_q_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_k_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_v_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_q_pre = Identity()
            self.layer_norm_k_pre = Identity()
            self.layer_norm_v_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_v_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_v_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_v_post = Identity()
        return

    def init_parameters(self):
        bound = (3 / self.emb_size) ** 0.5
        self.linear_q.weight.uniform_(-bound, bound)
        self.linear_q.bias.uniform_(-bound, bound)
        self.linear_k.weight.uniform_(-bound, bound)
        self.linear_k.bias.uniform_(-bound, bound)
        self.linear_v.weight.uniform_(-bound, bound)
        self.linear_v.bias.uniform_(-bound, bound)
        self.linear_out.weight.uniform_(-bound, bound)
        self.linear_out.bias.uniform_(-bound, bound)

        return

    def forward(self, input_q, input_k, input_v, input_mask):
        normalized_q = self.layer_norm_q_pre(input_q)
        normalized_k = self.layer_norm_k_pre(input_k)
        normalized_v = self.layer_norm_v_pre(input_v)

        length_in = input_k.size(1)
        batch_size, length_out, _ = input_q.size()

        query = self.linear_q(normalized_q).view(batch_size, length_out, self.num_of_heads, -1).contiguous().\
            transpose(1, 2).contiguous()
        key = self.linear_k(normalized_k).view(batch_size, length_in, self.num_of_heads, -1).contiguous().\
            transpose(1, 2).contiguous()
        value = self.linear_v(normalized_v).view(batch_size, length_in, self.num_of_heads, -1).contiguous().\
            transpose(1, 2).contiguous()

        alignments = query.matmul(key.transpose(-2, -1).contiguous()) / self.attention_norm_factor
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))

        context_vector = alignment_scores.matmul(value).transpose(1, 2).contiguous(). \
            view(batch_size, length_out, -1).contiguous()
        output = self.dropout_residual(self.linear_out(context_vector))
        return self.layer_norm_v_post(output + input_q)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
