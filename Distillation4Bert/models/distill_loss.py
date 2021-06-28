import torch.nn.functional as F
import torch
from typing import List

def cos_loss(state_S, state_T, mask=None):
    '''
    This is the loss used in DistilBERT

    :param state_S: Tensor of shape  (batch_size, length, hidden_size)
    :param state_T: Tensor of shape  (batch_size, length, hidden_size)
    :param mask:    Tensor of shape  (batch_size, length)
    '''
    if mask is  None:
        state_s = state_S.view(-1,state_S.size(-1))
        state_t = state_T.view(-1,state_T.size(-1))
    else:
        hid = state_S.size(-1)
        state_s = torch.masked_select(state_S, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_s = state_s.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        state_t = torch.masked_select(state_T, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_t = state_t.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert state_s.size() == state_t.size()

    target = state_s.new(state_s.size(0)).fill_(1)
    loss = F.cosine_embedding_loss(state_s, state_t, target, reduction='mean')
    return loss


def pkd_loss(state_S, state_T, mask=None):
    '''
    This is the loss used in BERT-PKD

    :param state_S: Tensor of shape  (batch_size, length, hidden_size)
    :param state_T: Tensor of shape  (batch_size, length, hidden_size)
    '''

    cls_T = state_T[:,0] # (batch_size, hidden_dim)
    cls_S = state_S[:,0] # (batch_size, hidden_dim)
    normed_cls_T = cls_T/torch.norm(cls_T,dim=1,keepdim=True)
    normed_cls_S = cls_S/torch.norm(cls_S,dim=1,keepdim=True)
    loss = (normed_cls_S - normed_cls_T).pow(2).sum(dim=-1).mean()
    return loss


def mse_loss(state_s, state_T, mask = None):
    if mask is None:
        loss = F.mse_loss(state_s, state_T)
    else:
        if mask.type() == 'torch.LongTensor' or mask.type() == 'torch.cuda.LongTensor':
            mask = mask ==1
        hid = state_s.size(-1)
        state_s = torch.masked_select(state_s, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_s = state_s.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        state_T = torch.masked_select(state_T, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_T = state_T.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert state_s.size() == state_T.size()

        loss = F.mse_loss(state_s, state_T)
    return loss

def kl_loss(state_s, state_T, mask = None, temperature = 2.0):
    if mask is not None:
        hid = state_s.size(-1)
        state_s = torch.masked_select(state_s, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_s = state_s.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        state_T = torch.masked_select(state_T, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        state_T = state_T.view(-1, hid)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert state_s.size() == state_T.size()
    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    loss_ce = (
            loss_fct(
                F.log_softmax(state_s / temperature, dim=-1),
                F.softmax(state_T / temperature, dim=-1),
            )
            * (temperature) ** 2
    )
    return loss_ce