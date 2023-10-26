import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import linecache
from utils import collate_tokens


class LanguageDataset(Dataset):

    def __init__(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)

    def __getitem__(self, item):
        return self.tokenizer.encode(self.data[item])

    def __len__(self):
        return len(self.data)


class LanguagePairDataset(Dataset):

    def __init__(self, cfg):

        self.prob = cfg.mask_probability

        with open(cfg.data + 'train.en', 'r', encoding='utf-8') as f:
            self.src = f.readlines()

        with open(cfg.data + 'train.de', 'r', encoding='utf-8') as f:
            self.tgt = f.readlines()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)

    def __getitem__(self, item):
        return self.tokenizer.encode(self.src[item].strip(), return_tensors='pt'), \
               self.tokenizer.encode(self.tgt[item].strip(), return_tensors='pt')

    def __len__(self):
        return len(self.src)

    def collate_fn(self, samples):

        src_list = [s[0].squeeze() for s in samples]
        tgt_list = [s[1].squeeze() for s in samples]

        batch_list = src_list + tgt_list
        # print(batch_list)
        batch = collate_tokens(
            batch_list,
            pad_idx=self.tokenizer.pad_token_id
        )
        inputs, labels = self.mask_tokens(batch)
        return {
            "inputs": inputs,
            "labels": labels,
        }

    def mask_tokens(self, batch):
        inputs = batch.clone()
        labels = batch.clone()
        # We sample a few tokens in each sequence for MLM training (with probability 'self.prob')
        probability_matrix = torch.full(labels.shape, self.prob)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.tokenizer.pad_token_id

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class LineCacheDataset(Dataset):

    def __init__(self, cfg):

        self.prob = cfg.mask_probability

        self.data = cfg.data
        self.src_file = "train.en"
        self.tgt_file = "train.de"
        self.num = self.get_file_length()
        # with open(cfg.data + 'train.en', 'r', encoding='utf-8') as f:
        #     self.src = f.readlines()

        # with open(cfg.data + 'train.de', 'r', encoding='utf-8') as f:
        #     self.tgt = f.readlines()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)

    def __getitem__(self, item):
        src_item = linecache.getline(self.data + self.src_file, lineno=item).strip()
        tgt_item = linecache.getline(self.data + self.tgt_file, lineno=item).strip()
        return self.tokenizer.encode(src_item, return_tensors='pt'), \
               self.tokenizer.encode(tgt_item, return_tensors='pt')
    
    def get_file_length(self):
        i = 0
        for i, line in enumerate(open(self.data + self.src_file, 'r')):
            pass
        return i + 1

    def __len__(self):
        return self.num

    def collate_fn(self, samples):

        src_list = [s[0].squeeze() for s in samples]
        tgt_list = [s[1].squeeze() for s in samples]

        batch_list = src_list + tgt_list
        # print(batch_list)
        batch = collate_tokens(
            batch_list,
            pad_idx=self.tokenizer.pad_token_id
        )
        inputs, labels = self.mask_tokens(batch)
        return {
            "inputs": inputs,
            "labels": labels,
        }

    def mask_tokens(self, batch):
        inputs = batch.clone()
        labels = batch.clone()
        # We sample a few tokens in each sequence for MLM training (with probability 'self.prob')
        probability_matrix = torch.full(labels.shape, self.prob)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.tokenizer.pad_token_id

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels