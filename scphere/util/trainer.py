import torch
from torch import nn
from scphere.util.data import DataSet

# ==============================================================================
class Trainer(object):
    def __init__(self, x, model, batch_id=None, mb_size=128,
                 learning_rate=0.001, max_epoch=100, depth_loss=False):

        self.model, self.mb_size, self.max_epoch, self.depth_loss, self.batch_id= \
            model, mb_size, max_epoch, depth_loss, batch_id

        self.max_iter = int(x.shape[0] / self.mb_size) * \
            self.max_epoch

        if batch_id is None:
            self.x = DataSet(x)
        else:
            self.x = DataSet(x, batch_id)

        enc_decay_params, enc_no_decay_params, \
        dec_decay_params, dec_no_decay_params = self.get_decay_params()

        self.optimizer = torch.optim.AdamW(
            [{'params': enc_decay_params, 'weight_decay': 0.01},
             {'params': enc_no_decay_params, 'weight_decay': 0},
             {'params': dec_decay_params, 'weight_decay': 0.01},
             {'params': dec_no_decay_params, 'weight_decay': 0},
             {'params': self.model.z_mu_layer.parameters(), 'weight_decay': 0},
             {'params': self.model.z_sigma_square_layer.parameters(), 'weight_decay': 0},
             {'params': self.model.mu_layer.parameters(), 'weight_decay': 0},
             {'params': self.model.dispersion, 'weight_decay': 0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08, # for regular Adam, eps=0.01
        )

        self.status = dict()
        self.status['kl_divergence'] = []
        self.status['log_likelihood'] = []
        self.status['elbo'] = []

        print("Finished creating Optimizer!")

    def train(self):
        self.model.train()
        for iter_i in range(self.max_iter):
            self.optimizer.zero_grad()

            x_mb, y_mb = self.x.next_batch(self.mb_size)
            x_mb, y_mb = torch.tensor(x_mb).to(self.model.device), torch.tensor(y_mb).to(self.model.device)

            self.model.forward(x_mb, y_mb)

            loss = -self.model.ELBO(x_mb)
            if self.depth_loss & (self.model.observation_dist == 'nb'):
                loss += self.model.depth_regularizer(x_mb, self.model.batch)

            loss.backward()

            # nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
            # nn.utils.clip_grad_value_(self.model.parameters(), 15)
            self.optimizer.step()

            if (iter_i % 50) == 0:
                ll = self.model.log_likelihood.detach().cpu().numpy()
                kl = self.model.kl.detach().cpu().numpy()
                elbo = loss.detach().cpu().numpy()
                self.status['log_likelihood'].append(ll)
                self.status['kl_divergence'].append(kl)
                self.status['elbo'].append(elbo)

                info_print = {'Log-likelihood': ll,
                              'ELBO': elbo, 'KL': kl}
                print(iter_i, '/', self.max_iter, info_print)

    @torch.no_grad()
    def get_decay_params(self):
        enc_decay_params = list()
        enc_no_decay_params = list()

        for name, param in self.model.encoder.named_parameters():
            if hasattr(param, 'requires_grad') and not param.requires_grad:
                continue
            if 'weight' in name and 'norm' not in name and 'bn' not in name:
                enc_decay_params.append(param)
            else:
                enc_no_decay_params.append(param)

        dec_decay_params = list()
        dec_no_decay_params = list()

        for name, param in self.model.decoder.named_parameters():
            if hasattr(param, 'requires_grad') and not param.requires_grad:
                continue
            if 'weight' in name and 'norm' not in name and 'bn' not in name:
                dec_decay_params.append(param)
            else:
                dec_no_decay_params.append(param)
        
        return enc_decay_params, enc_no_decay_params, dec_decay_params, dec_no_decay_params