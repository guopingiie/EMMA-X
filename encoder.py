from transformers import RobertaPreTrainedModel, RobertaModel, XLMRobertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from packaging import version


class EMMAEncoder(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    # _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias", r"dense"]

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, do_mlm=False):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.do_mlm = do_mlm
        if do_mlm:
            self.lm_head = RobertaLMHead(config)
            self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        sentence_output = self.dense(pooled_output)

        prediction_scores = None
        if self.do_mlm:
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

        # sentence_output = sequence_output[:, 0]

        return prediction_scores, F.normalize(sentence_output, p=2, dim=1)


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.gelu = GELUActivation()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias