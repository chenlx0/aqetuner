from module import *
# from encoder.tcnn import TCNNEncoder
from pair_encoder import AttnEncoder, TreeEncoder, NODE_DIM, HIDDEN_DIM
from torch.distributions import kl_divergence, Normal

class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, num_hidden):
        super(LatentModel, self).__init__()
        # self.pair_encoder = TreeEncoder(16, 16)
        self.pair_encoder = AttnEncoder(NODE_DIM, 16)
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)
        self.decoder_l = Decoder(num_hidden)
        self.decoder_e = Decoder(num_hidden)


    def forward(self, context_pairs, context_y, target_pairs, target_y=None):
        context_yl = context_y[:, 0].unsqueeze(-1)
        context_ye = context_y[:, 1].unsqueeze(-1)
        context_x = self.pair_encoder(context_pairs)
        target_x = self.pair_encoder(target_pairs)
        target_xl, target_xe = target_x, target_x
        cond = context_ye == 0.
        cond = cond.all(1)
        context_xl = context_x[cond, :]
        context_yl = context_yl[cond, :]
        context_xe = context_x
        num_targets = target_x.size(0)
        
        prior_mu, prior_var, z, vl, ve = self.latent_encoder(context_xl, context_xe, context_yl, context_ye)
        
        if target_y is not None:
            target_yl = target_y[:, 0].unsqueeze(-1)
            target_ye = target_y[:, 1].unsqueeze(-1)
            cond = target_ye == 0.
            cond = cond.all(1)
            target_xl = target_x[cond, :]
            target_yl = target_yl[cond, :]
            target_xe = target_x
            posterior_mu, posterior_var, z, vl, ve = self.latent_encoder(target_xl, target_xe, target_yl, target_ye)
        
        z = z.repeat(num_targets,1) # [B, T_target, H]
        rl, re = self.deterministic_encoder(context_xl, context_xe, context_yl, context_ye, target_xl, target_xe) # [B, T_target, H]
        
        # mu should be the prediction of target y
        vl = vl.repeat(target_xl.size(0), 1)
        ve = ve.repeat(target_xe.size(0), 1)
        y_pred_l = self.decoder_l(rl, vl, target_xl)
        y_pred_e = self.decoder_e(re, ve, target_xe, is_bernoulli=True)
        
        # For Training
        if target_y is not None:
            # get log probability
            log_likelihood = y_pred_l.log_prob(target_yl).mean(dim=0).sum()
            log_likelihood += y_pred_e.log_prob(target_ye).mean(dim=0).sum()
            
            # get KL divergence between prior and posterior
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
            # maximize prob and minimize KL divergence
            loss = -log_likelihood + kl
        # For Generation
        else:
            kl = None
            loss = None
        
        return y_pred_l, y_pred_e, loss
    
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div
