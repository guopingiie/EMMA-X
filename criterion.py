from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class EMContrastive(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.eps = args.label_smoothing
        self.rank_conf = args.rank_confidence
        self.rank_num = args.rank_num
        self.balance = args.loss_balance
        self.low_temp = args.low_t
        self.high_temp = args.high_t
        self.pad = AutoTokenizer.from_pretrained(args.pretrained_model_path).pad_token_id
        self.do_mlm = args.do_mlm

    def forward(self, logits, mlm_labels=None):
        """
        mlm_logits   --->   B x len x vocab
        mlm_labels   --->   B x len
        gmm_logits   --->   B x Q x rank_num
        gmm_labels   --->   B x Q
        sample_sim   --->   B x Q
        sample_rank   --->   B x Q
        """

        overall_loss_dict = {}
        total_loss = 0.0
        total_cts_loss = 0.0
        # const_loss_list = []

        # Calculate rank_num - 1 contrastive loss terms
        if logits.get("cts", None) is not None:

            cts = logits["cts"]
            bs = cts["logits"].size(0)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(cts["logits"]),
                1,
                torch.arange(bs).view(-1, 1).to(cts["logits"].device),
                0
            )
            
            temp_list = self.set_each_temp()
            for i in range(1, self.rank_num):
                # print('this is rank', i)
                rank_mask = cts["label"].eq(i).float().to(cts["label"].device)
                rank_mask = rank_mask * logits_mask
                if rank_mask.int().sum() == 0:
                    # const_loss_list.append(0.0)
                    continue
                negative_mask = cts["label"].le(i).float().to(cts["label"].device)

                negative_mask = logits_mask * negative_mask

                loss = self.calculate_contrastive_loss(logits=cts["logits"],
                                                    neg_mask=negative_mask,
                                                    pos_mask=rank_mask,
                                                    temp=temp_list[i - 1],
                                                    )
                total_cts_loss = total_cts_loss + loss
                # const_loss_list.append(loss.item())
            overall_loss_dict["cts"] = total_cts_loss
            total_loss = total_loss + total_cts_loss

        if logits.get("gmm", None) is not None:
            gmm = logits["gmm"]
            gmm_loss = self.calculate_label_smoothed_ce_loss(gmm["logits"],
                                                                gmm["label"],
                                                                eps=self.rank_conf)
            total_loss = total_loss + gmm_loss
            overall_loss_dict["gmm"] = gmm_loss

        if logits.get("mlm", None) is not None:
            mlm = logits["mlm"]
            assert mlm_labels is not None
            mlm_loss = self.calculate_label_smoothed_ce_loss(mlm["logits"],
                                                            mlm_labels.to(mlm["logits"].device),
                                                            eps=self.eps,
                                                            ignore_idx=self.pad)
            total_loss = total_loss + mlm_loss
            overall_loss_dict["mlm"] = mlm_loss
        overall_loss_dict["total"] = total_loss
        # print(total_const_loss, gaussian_loss)
        # loss = total_cts_loss + mlm_loss + self.balance * gmm_loss
        return overall_loss_dict

    def calculate_contrastive_loss(self, logits, neg_mask=None, pos_mask=None, temp=1):

        # SupCon loss (follow https://github.com/HobbitLong/SupContrast/blob/master/losses.py)
        curr_logits = logits / temp
        # print('logits: ', logits)
        exp_logits = torch.exp(curr_logits)
        if neg_mask is not None:
            exp_logits = exp_logits * neg_mask
        prob = exp_logits / (exp_logits + 1e-7).sum(1, keepdim=True)
        # prob = torch.softmax()
        # print('log_prob: ', log_prob)
        mean_prob_pos = (pos_mask.float() * prob).sum(-1) / (pos_mask + 1e-7).float().sum(-1)
        # print('mean log prob pos: ', mean_log_prob_pos)

        loss = - torch.log(mean_prob_pos + 1e-7).mean()
        # loss = loss.mean()
        # print('loss: ', loss)

        return loss

    def calculate_label_smoothed_ce_loss(self, logits, labels, eps, ignore_idx=None):
        # print(logits)
        lprob = torch.log_softmax(logits, dim=-1)
        # print('---------')
        # print(lprob)
        labels = labels.unsqueeze(dim=-1)
        nll_loss = -lprob.gather(dim=-1, index=labels)
        smooth_loss = -lprob.sum(dim=-1, keepdim=True)
        num = labels.numel()
        if ignore_idx is not None:
            ignore_mask = labels.eq(ignore_idx)
            nll_loss.masked_fill_(ignore_mask, 0.0)
            smooth_loss.masked_fill_(ignore_mask, 0.0)
            num = (~ignore_mask).sum()
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()


        eps_i = eps / (logits.size(-1) - 1)
        loss = (1.0 - eps - eps_i) * nll_loss + eps_i * smooth_loss
        loss = loss / (num + 1e-12)
        return loss

    def set_each_temp(self):
        temp_gap = (self.high_temp - self.low_temp) / (self.rank_num - 2)
        temp_list = [self.high_temp - temp_gap * i for i in range(self.rank_num - 1)]
        return temp_list




