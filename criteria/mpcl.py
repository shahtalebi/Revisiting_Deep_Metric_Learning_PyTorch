import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS = ['npair']
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
            cp = cos1(a_embs,p_embs)
            # cn = torch.zeros(a_embs.shape[0])
            cn = cos1(a_embs,n_embs[0, None,:])
            for k in range(1,n_embs.shape[0]):
                cn += cos1(a_embs,n_embs[k, None,:])
            cn = cn / n_embs.shape[0]
            loss = (cp - cn - 1).pow(2)
            loss = loss[:,None]

        return loss
