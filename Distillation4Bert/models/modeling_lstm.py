
import torch
from torch import nn

from transformers import BertPreTrainedModel, PretrainedConfig,BertConfig

import logging
import os
logger =  logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BertLayerNorm = torch.nn.LayerNorm

LSTM_MODEL_NAME = 'pytorch_model.bin'

class LSTMEncoder(nn.Module):
    def __init__(self, config, fit_size = 768):
        super(LSTMEncoder, self).__init__()
        self.layer = nn.ModuleList([nn.LSTM(
            config.hidden_size,
            config.hidden_size//2,
            bidirectional=True,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
            for _ in range(config.num_hidden_layers)])
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.lstm_layerNorm = BertLayerNorm( config.hidden_size, eps=1e-12)
        self.lstm_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        all_encoder_layers = ()
        all_encoder_atts = ()
        inputs = hidden_states
        hidden = None
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers = all_encoder_layers + (inputs,)
            inputs, hidden = layer_module(
                inputs, hidden)

            inputs = self.lstm_layerNorm(inputs)
            inputs = self.lstm_dropout(inputs)
            hidden = (self.lstm_dropout(hidden[0]), self.lstm_dropout(hidden[1]))

            all_encoder_atts = all_encoder_atts + (hidden[0], )

        all_encoder_layers = all_encoder_layers + (inputs,)

        tmp = []
        for layer in all_encoder_layers:
            tmp.append( nn.Tanh()(self.fit_dense(inputs)))
        all_encoder_layers = tmp

        return all_encoder_layers[-1], all_encoder_layers, all_encoder_atts

class LSTMBaseModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        # Save config in model
        self.config = config

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, LSTM_MODEL_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):

        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        resolved_archive_file = None
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, LSTM_MODEL_NAME)):
                    # Load from a PyTorch checkpoint
                    resolved_archive_file = os.path.join(pretrained_model_name_or_path, LSTM_MODEL_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {LSTM_MODEL_NAME} found in directory {pretrained_model_name_or_path} "
                        )
            elif os.path.isfile(pretrained_model_name_or_path):
                resolved_archive_file = pretrained_model_name_or_path

        logger.info("loading weights file {}".format(resolved_archive_file))
        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        model.load_state_dict(state_dict)

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        return model

class BLstmModel(LSTMBaseModel):
    def __init__(self, config, is_student = True ):
        super(BLstmModel, self).__init__(config)
        self.embedding =  nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, output_att=True):

        embedding_output = self.embedding(input_ids)
        embedding_output = self.LayerNorm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        output, lstm_sequence_output, lstm_hidden = self.lstm_encoder(embedding_output)

        return output, lstm_sequence_output, lstm_hidden


class BLSTMForSequenceClassification(LSTMBaseModel):
    def __init__(self, config, is_student = True, fit_size = 768):
        super(BLSTMForSequenceClassification, self).__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.lstm_encoder = LSTMEncoder(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_labels = config.num_labels
        self.classifier = nn.Linear(fit_size, config.num_labels)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, is_student=False):

        embedding_output = self.embedding(input_ids)
        embedding_output = self.LayerNorm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        output, lstm_sequence_output, lstm_hidden = self.lstm_encoder(embedding_output)

        #hidden_state = torch.cat((lstm_hidden[-1][0], lstm_hidden[-1][1]), -1)

        logits = self.classifier(output[:,-1,:])

        return logits, lstm_sequence_output, lstm_hidden