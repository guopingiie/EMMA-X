from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils import concat_all_gather
import numpy as np
from encoder import EMMAEncoder


class EMMA(nn.Module):

    def __init__(self, args):
        super().__init__()
        cfg = AutoConfig.from_pretrained(args.pretrained_model_path)
        self.pad = AutoTokenizer.from_pretrained(args.pretrained_model_path).pad_token_id
        self.encoder_q = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=args.do_mlm)
        self.encoder_k = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=args.do_mlm)
        self.K = args.K
        self.m = args.m
        # self.temperature = cfg.temperature
        self.rank_num = args.rank_num
        # self.rank_mean = torch.zeros(args.rank_num)
        self.update_mean = args.update_mean
        self.print_freq = args.print_freq
        self.warm_up = args.warm_up

        # Build Gaussian Mixture Module
        self.mu = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.lg_sigma2 = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.pi = Parameter(torch.ones(args.rank_num))
        # self.classifier = nn.Linear(cfg.hidden_size * 2, args.rank_num)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.K, cfg.hidden_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("rank_mean", torch.zeros(args.rank_num))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, inputs, is_pair=True):

        padding_mask = inputs.ne(self.pad)
        logits_dict = {}

        mlm_logits, sent_q = self.encoder_q(
            input_ids=inputs,
            attention_mask=padding_mask,
        )

        # mlm parts
        if mlm_logits is not None:
            mlm_pred = torch.argmax(mlm_logits, dim=-1)
            logits_dict["mlm"] = {"logits": mlm_logits, "pred": mlm_pred}

        with torch.no_grad():
            self._momentum_update_key_encoder()

            _, sent_k = self.encoder_k(
                input_ids=inputs,
                attention_mask=padding_mask,
                # output_hidden_states=True
            )

        all_feature = torch.cat([sent_k, self.queue.clone().detach()], dim=0)
        bs = sent_q.size(0)

        golden_mask = None
        if is_pair:
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            rev_q = pair_label.view(-1, 1)
            golden_labels = torch.eq(torch.arange(bs).view(-1, 1), rev_q.T)
            golden_mask = torch.cat([golden_labels, torch.zeros(sent_q.size(0), self.queue.size(0))], dim=-1).to(
                sent_q.device).bool()

        # similarity calculation
        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        # E-step: gmm part calculation
        sentence_q = sent_q.clone().detach()
        gmm_feature = sentence_q.unsqueeze(1) - all_feature.unsqueeze(0)
        gmm_label = cts_pred

        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
            
        self.update_rank_mean(concat_all_gather(sample_sim), concat_all_gather(gmm_pred))
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

        # M-step: ctl part calculation
        cts_label = gmm_pred
        if golden_mask is not None:
            cts_label.masked_fill_(golden_mask, self.rank_num - 1)

        # for numerical stability
        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}

        # dequeue and enqueue
        self._dequeue_and_enqueue(sent_k)

        return logits_dict

    def _calculate_ctl(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):
        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)

        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()

        cts_label = golden_mask * (self.rank_num - 1)

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

    def _calculate_gmm(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):
        sentence_q = sent_q.clone().detach()
        interploted_feature, gmm_feature, gmm_label = self.interploted_feature(sentence_q, all_feature)
        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)
        interploted_sample_sim = (sent_q.unsqueeze(1) * interploted_feature).sum(dim=-1)
        self.update_rank_mean(concat_all_gather(interploted_sample_sim), concat_all_gather(gmm_label))

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

    def cts_sim_dist(self, similarity):
        labels = torch.floor((similarity + 1) / 2 * self.rank_num)
        return labels.long()

    def covert_scores_to_ranks(self, scores):
        clip_scores = scores.masked_fill(scores.gt(1), 1)
        clip_scores.masked_fill_(scores.lt(0), 0)
        ranks = torch.zeros(scores.size(), device=scores.device)
        rank_bar = torch.softmax(self.pi, dim=-1)
        # print(rank_bar)
        rank_threshold = 0
        for i in range(self.rank_num):
            rank_mask = clip_scores.ge(rank_threshold) * clip_scores.lt(rank_threshold + rank_bar[i]).to(scores.device)
            rank_threshold += rank_bar[i]
            # print(rank_threshold)
            ranks.masked_fill_(rank_mask, i)
        # assert rank_threshold == 1
        return ranks.long()

    def interploted_feature(self, feature_q, feature_k):
        bs = feature_q.size(0)
        rev_feature_q = torch.cat([feature_q[bs // 2:, :], feature_q[:bs // 2, :]], dim=0)

        random_selection = torch.rand(feature_q.size(0), feature_k.size(0), device=feature_q.device).unsqueeze(-1)

        interploted_feature = random_selection * rev_feature_q.unsqueeze(1) \
                              + (1 - random_selection) * feature_k.unsqueeze(0)

        # gmm_feature = torch.cat([feature_q.unsqueeze(1).expand(-1, feature_k.size(0), -1),
        #                             interploted_feature], dim=-1)
        gmm_feature = feature_q.unsqueeze(1) - interploted_feature
        interploted_label = self.covert_scores_to_ranks(random_selection.squeeze(-1))
        return interploted_feature, gmm_feature, interploted_label

    def map_scores_to_ranks(self, scores):
        rank_logits = torch.abs(
            scores.unsqueeze(dim=-1) - self.rank_mean.clone().detach().unsqueeze(dim=0).unsqueeze(dim=0))
        ranks = torch.argmin(rank_logits, dim=-1)
        return ranks.long()

    def update_rank_mean(self, cur_sim, cur_rank):
        # expand_sim = cur_sim.unsqueeze(dim=-1)
        rank_sim_mask = F.one_hot(cur_rank, num_classes=self.rank_num)
        select_sim = rank_sim_mask * cur_sim.unsqueeze(dim=-1)

        update_mask = rank_sim_mask.sum(dim=0).sum(dim=0).ne(0)

        cur_rank_mean = select_sim.sum(dim=0).sum(dim=0) / (rank_sim_mask.sum(dim=0).sum(dim=0).float() + 1e-12)
        self.rank_mean = (1 - update_mask * self.update_mean) * self.rank_mean \
                         + update_mask * self.update_mean * cur_rank_mean

    def gaussian_mixture_module(self, features):
        """
        input --> B x Q x E
        output --> B x Q x G
        mu   --->   G x E
        sigma   --->   G x E
        pi   --->   G
        """
        # # print('pi: ', self.pi)
        # # print('sigma: ', self.sigma.mean(dim=-1))
        # diffs = features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0)
        # # print('diffs: ', diffs.mean(dim=-1))
        # expo = torch.exp(-0.5 * (diffs.pow(2) / self.sigma.unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        # # print('expo: ', expo)
        # expo = expo / torch.sqrt(self.sigma.prod(dim=-1) + 1e-9).unsqueeze(0).unsqueeze(0)
        # weighted = torch.softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) * expo
        # ln2piD --> 1
        # ln2piD = torch.tensor(np.log(2 * np.pi) * self.embed_dim, device=features.device, requires_grad=False).float()
        # # diffs --> B x G x E
        # diffs = features.unsqueeze(1) - self.mu.unsqueeze(0)
        # # expo --> B x G
        # expo = (diffs.pow(2) / self.sigma.unsqueeze(0)).sum(dim=2)
        # # log_cof --> 1 x G
        # log_cof = torch.tensor(ln2piD + torch.log(self.sigma.prod(dim=-1) + 1e-9)).unsqueeze(dim=0)
        # log_components = -0.5 * (log_cof + expo)
        # # log_weighted --> B x G
        # log_weighted = log_components + torch.log_softmax(self.pi, dim=0).unsqueeze(dim=0)

        lg2piD = torch.tensor(features.size(-1) * np.log(2 * np.pi), device=features.device, requires_grad=False)

        diffs = (features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0))

        sigma2 = torch.exp(self.lg_sigma2).unsqueeze(0).unsqueeze(0)

        expo = (diffs.pow(2) / sigma2).sum(dim=-1)

        l_prob = -0.5 * (lg2piD + self.lg_sigma2.sum(dim=-1) + expo)

        logits = torch.log_softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) + l_prob

        logits_shift = logits.max(dim=-1, keepdim=True)[0]
        logits_normalize = logits - logits_shift
        pred = torch.softmax(logits_normalize, dim=-1)
        #
        # # # Re-normalize
        # # log_shift = log_weighted.max(dim=reduce_dim, keepdim=True)[0]
        # # # print(log_shift.size())
        # # exp_log_shifted = torch.exp(log_weighted - log_shift)
        # # exp_log_shifted_sum = exp_log_shifted.sum(dim=reduce_dim, keepdim=True)
        # # gamma = exp_log_shifted / exp_log_shifted_sum

        return logits_normalize, pred

    def init_guassian_module(self):
        pass


class EMMA_CTL(nn.Module):

    def __init__(self, args):
        super().__init__()
        cfg = AutoConfig.from_pretrained(args.pretrained_model_path)
        self.pad = AutoTokenizer.from_pretrained(args.pretrained_model_path).pad_token_id
        self.encoder_q = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=args.do_mlm)
        self.encoder_k = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=args.do_mlm)
        self.K = args.K
        self.m = args.m
        # self.temperature = cfg.temperature
        self.rank_num = args.rank_num
        # self.rank_mean = torch.zeros(args.rank_num)
        self.update_mean = args.update_mean
        self.print_freq = args.print_freq
        self.warm_up = args.warm_up

        # Build Gaussian Mixture Module
        self.mu = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.lg_sigma2 = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.pi = Parameter(torch.ones(args.rank_num))
        # self.classifier = nn.Linear(cfg.hidden_size * 2, args.rank_num)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.K, cfg.hidden_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("rank_mean", torch.zeros(args.rank_num))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, inputs, is_pair=True):

        padding_mask = inputs.ne(self.pad)
        logits_dict = {}

        mlm_logits, sent_q = self.encoder_q(
            input_ids=inputs,
            attention_mask=padding_mask,
        )

        # mlm parts
        if mlm_logits is not None:
            mlm_pred = torch.argmax(mlm_logits, dim=-1)
            logits_dict["mlm"] = {"logits": mlm_logits, "pred": mlm_pred}

        with torch.no_grad():
            self._momentum_update_key_encoder()

            _, sent_k = self.encoder_k(
                input_ids=inputs,
                attention_mask=padding_mask,
            )

        all_feature = torch.cat([sent_k, self.queue.clone().detach()], dim=0)
        bs = sent_q.size(0)

        golden_mask = None
        if is_pair:
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            rev_q = pair_label.view(-1, 1)
            golden_labels = torch.eq(torch.arange(bs).view(-1, 1), rev_q.T)
            golden_mask = torch.cat([golden_labels, torch.zeros(sent_q.size(0), self.queue.size(0))], dim=-1).to(
                sent_q.device).bool()

        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)

        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()

        cts_label = golden_mask * (self.rank_num - 1)

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

        # dequeue and enqueue
        self._dequeue_and_enqueue(sent_k)

        return logits_dict


    def _calculate_gmm(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):
        sentence_q = sent_q.clone().detach()
        interploted_feature, gmm_feature, gmm_label = self.interploted_feature(sentence_q, all_feature)
        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)
        interploted_sample_sim = (sent_q.unsqueeze(1) * interploted_feature).sum(dim=-1)
        self.update_rank_mean(concat_all_gather(interploted_sample_sim), concat_all_gather(gmm_label))

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

    def _iter_gmm_ctl(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):

        # similarity calculation
        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        # E-step: gmm part calculation
        sentence_q = sent_q.clone().detach()
        gmm_feature = sentence_q.unsqueeze(1) - all_feature.unsqueeze(0)
        gmm_label = cts_pred

        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)

        self.update_rank_mean(concat_all_gather(sample_sim), concat_all_gather(gmm_pred))

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

        # M-step: ctl part calculation
        cts_label = gmm_pred
        if golden_mask is not None:
            cts_label.masked_fill_(golden_mask, self.rank_num - 1)

        # for numerical stability
        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}

    def cts_sim_dist(self, similarity):
        labels = torch.floor((similarity + 1) / 2 * self.rank_num)
        return labels.long()

    def covert_scores_to_ranks(self, scores):
        clip_scores = scores.masked_fill(scores.gt(1), 1)
        clip_scores.masked_fill_(scores.lt(0), 0)
        ranks = torch.zeros(scores.size(), device=scores.device)
        rank_bar = torch.softmax(self.pi, dim=-1)
        # print(rank_bar)
        rank_threshold = 0
        for i in range(self.rank_num):
            rank_mask = clip_scores.ge(rank_threshold) * clip_scores.lt(rank_threshold + rank_bar[i]).to(scores.device)
            rank_threshold += rank_bar[i]
            # print(rank_threshold)
            ranks.masked_fill_(rank_mask, i)
        # assert rank_threshold == 1
        return ranks.long()

    def interploted_feature(self, feature_q, feature_k):
        bs = feature_q.size(0)
        rev_feature_q = torch.cat([feature_q[bs // 2:, :], feature_q[:bs // 2, :]], dim=0)

        random_selection = torch.rand(feature_q.size(0), feature_k.size(0), device=feature_q.device).unsqueeze(-1)

        interploted_feature = random_selection * rev_feature_q.unsqueeze(1) \
                              + (1 - random_selection) * feature_k.unsqueeze(0)

        # gmm_feature = torch.cat([feature_q.unsqueeze(1).expand(-1, feature_k.size(0), -1),
        #                             interploted_feature], dim=-1)
        gmm_feature = feature_q.unsqueeze(1) - interploted_feature
        interploted_label = self.covert_scores_to_ranks(random_selection.squeeze(-1))
        return interploted_feature, gmm_feature, interploted_label

    def map_scores_to_ranks(self, scores):
        rank_logits = torch.abs(
            scores.unsqueeze(dim=-1) - self.rank_mean.clone().detach().unsqueeze(dim=0).unsqueeze(dim=0))
        ranks = torch.argmin(rank_logits, dim=-1)
        return ranks.long()

    def update_rank_mean(self, cur_sim, cur_rank):
        # expand_sim = cur_sim.unsqueeze(dim=-1)
        rank_sim_mask = F.one_hot(cur_rank, num_classes=self.rank_num)
        select_sim = rank_sim_mask * cur_sim.unsqueeze(dim=-1)

        update_mask = rank_sim_mask.sum(dim=0).sum(dim=0).ne(0)

        cur_rank_mean = select_sim.sum(dim=0).sum(dim=0) / (rank_sim_mask.sum(dim=0).sum(dim=0).float() + 1e-12)
        self.rank_mean = (1 - update_mask * self.update_mean) * self.rank_mean \
                         + update_mask * self.update_mean * cur_rank_mean

    def gaussian_mixture_module(self, features):
        """
        input --> B x Q x E
        output --> B x Q x G
        mu   --->   G x E
        sigma   --->   G x E
        pi   --->   G
        """
        # # print('pi: ', self.pi)
        # # print('sigma: ', self.sigma.mean(dim=-1))
        # diffs = features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0)
        # # print('diffs: ', diffs.mean(dim=-1))
        # expo = torch.exp(-0.5 * (diffs.pow(2) / self.sigma.unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        # # print('expo: ', expo)
        # expo = expo / torch.sqrt(self.sigma.prod(dim=-1) + 1e-9).unsqueeze(0).unsqueeze(0)
        # weighted = torch.softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) * expo
        # ln2piD --> 1
        # ln2piD = torch.tensor(np.log(2 * np.pi) * self.embed_dim, device=features.device, requires_grad=False).float()
        # # diffs --> B x G x E
        # diffs = features.unsqueeze(1) - self.mu.unsqueeze(0)
        # # expo --> B x G
        # expo = (diffs.pow(2) / self.sigma.unsqueeze(0)).sum(dim=2)
        # # log_cof --> 1 x G
        # log_cof = torch.tensor(ln2piD + torch.log(self.sigma.prod(dim=-1) + 1e-9)).unsqueeze(dim=0)
        # log_components = -0.5 * (log_cof + expo)
        # # log_weighted --> B x G
        # log_weighted = log_components + torch.log_softmax(self.pi, dim=0).unsqueeze(dim=0)

        lg2piD = torch.tensor(features.size(-1) * np.log(2 * np.pi), device=features.device, requires_grad=False)

        diffs = (features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0))

        sigma2 = torch.exp(self.lg_sigma2).unsqueeze(0).unsqueeze(0)

        expo = (diffs.pow(2) / sigma2).sum(dim=-1)

        l_prob = -0.5 * (lg2piD + self.lg_sigma2.sum(dim=-1) + expo)

        logits = torch.log_softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) + l_prob

        logits_shift = logits.max(dim=-1, keepdim=True)[0]
        logits_normalize = logits - logits_shift
        pred = torch.softmax(logits_normalize, dim=-1)
        #
        # # # Re-normalize
        # # log_shift = log_weighted.max(dim=reduce_dim, keepdim=True)[0]
        # # # print(log_shift.size())
        # # exp_log_shifted = torch.exp(log_weighted - log_shift)
        # # exp_log_shifted_sum = exp_log_shifted.sum(dim=reduce_dim, keepdim=True)
        # # gamma = exp_log_shifted / exp_log_shifted_sum

        return logits_normalize, pred

    def init_guassian_module(self):
        pass


class EMMA_GMM(nn.Module):

    def __init__(self, args):
        super().__init__()
        cfg = AutoConfig.from_pretrained(args.pretrained_model_path)
        self.pad = AutoTokenizer.from_pretrained(args.pretrained_model_path).pad_token_id
        self.encoder_q = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=False)
        self.encoder_k = EMMAEncoder.from_pretrained(args.pretrained_model_path, config=cfg, do_mlm=False)
        self.K = args.K
        self.m = args.m
        # self.temperature = cfg.temperature
        self.rank_num = args.rank_num
        # self.rank_mean = torch.zeros(args.rank_num)
        self.update_mean = args.update_mean
        self.print_freq = args.print_freq
        self.warm_up = args.warm_up

        # Build Gaussian Mixture Module
        self.mu = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.lg_sigma2 = Parameter(torch.randn(args.rank_num, cfg.hidden_size))
        self.pi = Parameter(torch.ones(args.rank_num))
        # self.classifier = nn.Linear(cfg.hidden_size * 2, args.rank_num)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        for param_q in self.encoder_q.parameters():
            param_q.requires_grad = False

        self.register_buffer("queue", torch.randn(self.K, cfg.hidden_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("rank_mean", torch.zeros(args.rank_num))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, inputs, is_pair=True):

        padding_mask = inputs.ne(self.pad)
        logits_dict = {}

        with torch.no_grad():

            _, sent_q = self.encoder_q(
                input_ids=inputs,
                attention_mask=padding_mask,
            )

            _, sent_k = self.encoder_k(
                input_ids=inputs,
                attention_mask=padding_mask,
            )

        all_feature = torch.cat([sent_k, self.queue.clone().detach()], dim=0)
        bs = sent_q.size(0)

        golden_mask = None
        if is_pair:
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            rev_q = pair_label.view(-1, 1)
            golden_labels = torch.eq(torch.arange(bs).view(-1, 1), rev_q.T)
            golden_mask = torch.cat([golden_labels, torch.zeros(sent_q.size(0), self.queue.size(0))], dim=-1).to(
                sent_q.device).bool()

        sentence_q = sent_q.clone().detach()
        interploted_feature, gmm_feature, gmm_label = self.interploted_feature(sentence_q, all_feature)
        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)
        interploted_sample_sim = (sent_q.unsqueeze(1) * interploted_feature).sum(dim=-1)

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        
        self.update_rank_mean(concat_all_gather(interploted_sample_sim), concat_all_gather(gmm_label))
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

        # dequeue and enqueue
        self._dequeue_and_enqueue(sent_k)

        return logits_dict

    def _calculate_ctl(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):
        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)

        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()

        cts_label = golden_mask * (self.rank_num - 1)

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

    def _calculate_gmm(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):
        sentence_q = sent_q.clone().detach()
        interploted_feature, gmm_feature, gmm_label = self.interploted_feature(sentence_q, all_feature)
        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)
        interploted_sample_sim = (sent_q.unsqueeze(1) * interploted_feature).sum(dim=-1)
        self.update_rank_mean(concat_all_gather(interploted_sample_sim), concat_all_gather(gmm_label))

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

    def _iter_gmm_ctl(self, sent_q, all_feature, logits_dict, is_pair=True, golden_mask=None):

        # similarity calculation
        sample_sim = torch.matmul(sent_q, all_feature.T)
        cts_pred = self.map_scores_to_ranks(sample_sim)
        cts_dist = self.cts_sim_dist(sample_sim)

        # E-step: gmm part calculation
        sentence_q = sent_q.clone().detach()
        gmm_feature = sentence_q.unsqueeze(1) - all_feature.unsqueeze(0)
        gmm_label = cts_pred

        gmm_logits, gmm_prob = self.gaussian_mixture_module(gmm_feature)
        gmm_pred = torch.argmax(gmm_prob, dim=-1)

        self.update_rank_mean(concat_all_gather(sample_sim), concat_all_gather(gmm_pred))

        if golden_mask is not None:
            gmm_mask = torch.eye(sent_q.size(0), all_feature.size(0)).to(golden_mask.device) + golden_mask
            gmm_label.masked_fill_(gmm_mask.bool(), self.rank_num - 1)
        logits_dict["gmm"] = {"logits": gmm_logits, "pred": gmm_pred, "label": gmm_label}

        # M-step: ctl part calculation
        cts_label = gmm_pred
        if golden_mask is not None:
            cts_label.masked_fill_(golden_mask, self.rank_num - 1)

        # for numerical stability
        logtis_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
        # logits  -->  bs x bs+queue
        cts_logits = sample_sim - logtis_max.detach()
        logits_dict["cts"] = {"logits": cts_logits, "pred": cts_pred, "label": cts_label, "dist": cts_dist}

        if is_pair:
            bs = sent_q.size(0)
            logits_mask = torch.scatter(
                torch.ones_like(sample_sim),
                1,
                torch.arange(bs).view(-1, 1).to(sample_sim.device),
                0
            )
            outputs = sample_sim * logits_mask
            pair_label = torch.cat([torch.arange(bs)[bs // 2:], torch.arange(bs)[:bs // 2]])
            logits_dict["acc"] = {"pred": outputs, "label": pair_label.to(sent_q.device)}

    def cts_sim_dist(self, similarity):
        labels = torch.floor((similarity + 1) / 2 * self.rank_num)
        return labels.long()

    def covert_scores_to_ranks(self, scores):
        clip_scores = scores.masked_fill(scores.gt(1), 1)
        clip_scores.masked_fill_(scores.lt(0), 0)
        ranks = torch.zeros(scores.size(), device=scores.device)
        rank_bar = torch.softmax(self.pi, dim=-1)
        # print(rank_bar)
        rank_threshold = 0
        for i in range(self.rank_num):
            rank_mask = clip_scores.ge(rank_threshold) * clip_scores.lt(rank_threshold + rank_bar[i]).to(scores.device)
            rank_threshold += rank_bar[i]
            # print(rank_threshold)
            ranks.masked_fill_(rank_mask, i)
        # assert rank_threshold == 1
        return ranks.long()

    def interploted_feature(self, feature_q, feature_k):
        # bs = feature_q.size(0)
        # rev_feature_q = torch.cat([feature_q[bs // 2:, :], feature_q[:bs // 2, :]], dim=0)

        random_selection = torch.rand(feature_q.size(0), feature_k.size(0), device=feature_q.device).unsqueeze(-1)

        interploted_feature = random_selection * feature_q.unsqueeze(1) \
                              + (1 - random_selection) * feature_k.unsqueeze(0)

        # gmm_feature = torch.cat([feature_q.unsqueeze(1).expand(-1, feature_k.size(0), -1),
        #                             interploted_feature], dim=-1)
        gmm_feature = feature_q.unsqueeze(1) - interploted_feature
        interploted_label = self.covert_scores_to_ranks(random_selection.squeeze(-1))
        return interploted_feature, gmm_feature, interploted_label

    def map_scores_to_ranks(self, scores):
        rank_logits = torch.abs(
            scores.unsqueeze(dim=-1) - self.rank_mean.clone().detach().unsqueeze(dim=0).unsqueeze(dim=0))
        ranks = torch.argmin(rank_logits, dim=-1)
        return ranks.long()

    def update_rank_mean(self, cur_sim, cur_rank):
        # expand_sim = cur_sim.unsqueeze(dim=-1)
        rank_sim_mask = F.one_hot(cur_rank, num_classes=self.rank_num)
        select_sim = rank_sim_mask * cur_sim.unsqueeze(dim=-1)

        update_mask = rank_sim_mask.sum(dim=0).sum(dim=0).ne(0)

        cur_rank_mean = select_sim.sum(dim=0).sum(dim=0) / (rank_sim_mask.sum(dim=0).sum(dim=0).float() + 1e-12)
        self.rank_mean = (1 - update_mask * self.update_mean) * self.rank_mean \
                         + update_mask * self.update_mean * cur_rank_mean

    def gaussian_mixture_module(self, features):
        """
        input --> B x Q x E
        output --> B x Q x G
        mu   --->   G x E
        sigma   --->   G x E
        pi   --->   G
        """
        # # print('pi: ', self.pi)
        # # print('sigma: ', self.sigma.mean(dim=-1))
        # diffs = features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0)
        # # print('diffs: ', diffs.mean(dim=-1))
        # expo = torch.exp(-0.5 * (diffs.pow(2) / self.sigma.unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        # # print('expo: ', expo)
        # expo = expo / torch.sqrt(self.sigma.prod(dim=-1) + 1e-9).unsqueeze(0).unsqueeze(0)
        # weighted = torch.softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) * expo
        # ln2piD --> 1
        # ln2piD = torch.tensor(np.log(2 * np.pi) * self.embed_dim, device=features.device, requires_grad=False).float()
        # # diffs --> B x G x E
        # diffs = features.unsqueeze(1) - self.mu.unsqueeze(0)
        # # expo --> B x G
        # expo = (diffs.pow(2) / self.sigma.unsqueeze(0)).sum(dim=2)
        # # log_cof --> 1 x G
        # log_cof = torch.tensor(ln2piD + torch.log(self.sigma.prod(dim=-1) + 1e-9)).unsqueeze(dim=0)
        # log_components = -0.5 * (log_cof + expo)
        # # log_weighted --> B x G
        # log_weighted = log_components + torch.log_softmax(self.pi, dim=0).unsqueeze(dim=0)

        lg2piD = torch.tensor(features.size(-1) * np.log(2 * np.pi), device=features.device, requires_grad=False)

        diffs = (features.unsqueeze(2) - self.mu.unsqueeze(0).unsqueeze(0))

        sigma2 = torch.exp(self.lg_sigma2).unsqueeze(0).unsqueeze(0)

        expo = (diffs.pow(2) / sigma2).sum(dim=-1)

        l_prob = -0.5 * (lg2piD + self.lg_sigma2.sum(dim=-1) + expo)

        logits = torch.log_softmax(self.pi, dim=0).unsqueeze(0).unsqueeze(0) + l_prob

        logits_shift = logits.max(dim=-1, keepdim=True)[0]
        logits_normalize = logits - logits_shift
        pred = torch.softmax(logits_normalize, dim=-1)
        #
        # # # Re-normalize
        # # log_shift = log_weighted.max(dim=reduce_dim, keepdim=True)[0]
        # # # print(log_shift.size())
        # # exp_log_shifted = torch.exp(log_weighted - log_shift)
        # # exp_log_shifted_sum = exp_log_shifted.sum(dim=reduce_dim, keepdim=True)
        # # gamma = exp_log_shifted / exp_log_shifted_sum

        return logits_normalize, pred

    def init_guassian_module(self):
        pass