
from transformers import BertPreTrainedModel, BertModel
from transformers.activations import gelu, gelu_new, swish
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
BertLayerNorm = torch.nn.LayerNorm

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class DistillBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config, is_student = False, fit_size = 768):
        super().__init__(config)

        config.output_hidden_states = True
        config.output_attentions = True

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.is_student = is_student
        if self.is_student:
            self.fit_dense = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        hidden_states, attentions = outputs[2:]

        if self.is_student:
            tmp = []
            for hid in hidden_states:
                tmp.append(self.fit_dense(hid))
            hidden_states = tmp

        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        outputs = (prediction_scores, hidden_states, attentions)


        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)

class DistillBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, is_student = False, fit_size = 768):
        super(DistillBertForSequenceClassification, self).__init__(config)
        config.output_hidden_states = True
        config.output_attentions = True

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.is_student = is_student
        if self.is_student:
            self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.num_labels = config.num_labels
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        hidden_states, attentions = outputs[2:]

        if self.is_student:
            tmp = []
            for hid in hidden_states:
                tmp.append(self.fit_dense(hid))
            hidden_states = tmp

        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        outputs = (logits, hidden_states, attentions)
        return outputs


