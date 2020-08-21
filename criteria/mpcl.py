import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS = ['mpcl']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
        """
        super(Criterion, self).__init__()
        self.pars = opt
        # self.l2_weight = opt.loss_npair_l2
        self.batchminer = batchminer

        self.name           = 'mpcl'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.batchminer(batch, labels)

        ##
        loss  = 0
        if 'bninception' in self.pars.arch:
            ### clamping/value reduction to avoid initial overflow for high embedding dimensions!
            batch = batch/4
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor+1], batch[positive:positive+1], batch[negative_set]
            cos1 = torch.nn.CosineSimilarity(dim=1)
            cos2 = torch.nn.CosineSimilarity(dim=2)
            cp = cos1(a_embs,p_embs)
            cn = [cos2(a_embs[:,None,:],n_embs[:,k:k+1,:]) for k in range(n_embs.shape[1])]
            loss = (cp[:,None] - sum(cn)/(n_embs.shape[1]) - 1)**2
            
            # inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            # inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            # loss  = loss + torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1))/len(anchors)
            # loss  = loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))/len(anchors)

        return loss.reshape(-1)
