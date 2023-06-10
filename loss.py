import torch
from torch.nn.modules.loss import _Loss

class PairwiseNegSDR(_Loss):
    EPS = 1e-8

    def forward(self, y, y_):
        # assert targets.size() == est_targets.size() and targets.ndim == 3, \
        #     f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
        s_target = torch.unsqueeze(y_ - torch.mean(y_, dim=2, keepdim=True), dim=1)
        if len(y_.shape) > len(y.shape):
            s_target = torch.mean(s_target, axis=0)
        s_estimate = torch.unsqueeze(y - torch.mean(y, dim=2, keepdim=True), dim=2)

        # [batch, n_src, n_src, 1]
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
        # [batch, 1, n_src, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
        # [batch, n_src, n_src, time]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy

        e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.EPS)
        return -10 * torch.log10(pair_wise_sdr + self.EPS)
    
    
class PITLoss(_Loss):
    
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, y, y_):
        if len(y.shape) > len(y_.shape):
            l = torch.mean([self.loss_func(x, y_) for x in y])
        l = self.loss_func(y, y_)
        return torch.mean(torch.min(l[:,0,0]+l[:,1,1], l[:,1,0]+l[:,0,1])/2)

PILPairwiseSDR = PITLoss(PairwiseNegSDR())


