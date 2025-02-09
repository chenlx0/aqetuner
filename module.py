import torch as t
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Normal, Bernoulli

QUERY_DIM=32

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, input_dim=QUERY_DIM+1):
        super(LatentEncoder, self).__init__()
        self.input_projection_e = Linear(input_dim, num_hidden)
        self.input_projection_l = Linear(input_dim, num_hidden)
        # self.self_attentions_e = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        # self.self_attentions_l = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, xl, xe, yl, ye):
        # concat location (x) and value (y)
        encoder_input_e = t.cat([xe, ye], dim=-1)
        encoder_input_l = t.cat([xl, yl], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input_e = self.input_projection_e(encoder_input_e)
        encoder_input_l = self.input_projection_l(encoder_input_l)
        
        # self attention layer
        # for attention in self.self_attentions:
            # encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # mean
        hidden_e = encoder_input_e.mean(dim=0)
        hidden_l = encoder_input_l.mean(dim=0)
        
        hidden_g = t.relu((hidden_l + hidden_e)/2)
        hidden_l = t.tanh(hidden_l) + t.sigmoid(hidden_g) * hidden_g
        hidden_e = t.tanh(hidden_e) + t.sigmoid(hidden_g) * hidden_e
        hidden_e = t.relu(self.penultimate_layer(hidden_e))
        hidden_l = t.relu(self.penultimate_layer(hidden_l))
        # get mu and sigma
        mu = self.mu(hidden_g)
        log_sigma = self.log_sigma(hidden_g)
        
        # reparameterization trick
        std = t.exp(0.5 * log_sigma)
        eps = t.randn_like(std)
        z = eps.mul(std).add_(mu)
        
        # return distribution
        return mu, log_sigma, z, hidden_l, hidden_e
    

class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """
    def __init__(self, num_hidden, num_latent, input_dim=QUERY_DIM+1):
        super(DeterministicEncoder, self).__init__()
        # self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions_l = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions_e = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(QUERY_DIM, num_hidden)
        self.target_projection = Linear(QUERY_DIM, num_hidden)

    def forward(self, context_xl, context_xe, context_yl, context_ye, target_xl, target_xe):
        # concat context location (x), context value (y)
        encoder_input_l = t.cat([context_xl,context_yl], dim=-1)
        encoder_input_e = t.cat([context_xe,context_ye], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input_l = self.input_projection(encoder_input_l)
        encoder_input_e = self.input_projection(encoder_input_e)
        
        # self attention layer
        # for attention in self.self_attentions:
            # encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # query: target_x, key: context_x, value: representation
        query_l = self.target_projection(target_xl)
        keys_l = self.context_projection(context_xl)
        query_e = self.target_projection(target_xe)
        keys_e = self.context_projection(context_xe)
        
        # cross attention layer
        for attention in self.cross_attentions_l:
            query_l, _ = attention(keys_l, encoder_input_l, query_l)
        for attention in self.cross_attentions_e:
            query_e, _ = attention(keys_e, encoder_input_e, query_e)
        return query_l, query_e
    

class Decoder(nn.Module):
    """
    Decoder for generation 
    """
    def __init__(self, num_hidden):
        super(Decoder, self).__init__()
        self.target_projection = Linear(QUERY_DIM, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        self.final_to_mu = Linear(num_hidden * 3, 1)
        self.final_to_sigma = Linear(num_hidden * 3, 1)
        
    def forward(self, r, z, target_x, is_bernoulli=False):
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)
        
        # concat all vectors (r,z,target_x)
        hidden = t.cat([t.cat([r,z], dim=-1), target_x], dim=-1)
        
        # mlp layers
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
            
        # get mu and sigma
        y_pred_mu = self.final_to_mu(hidden)
        if is_bernoulli:
            return Bernoulli(t.sigmoid(y_pred_mu))
        y_pred_sigma = self.final_to_sigma(hidden)
        y_pred_sigma = 0.1 + 0.9 * F.softplus(y_pred_sigma)
        return Normal(y_pred_mu, y_pred_sigma)


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(num_hidden, h, batch_first=True)

    def forward(self, key, value, query):
        result, attns = self.multihead_attention(query, key, value)
        return result, attns
