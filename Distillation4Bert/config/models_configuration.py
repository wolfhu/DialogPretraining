

import torch
import os
import logging
import copy
import json

logger = logging.getLogger(__name__)

class DistillConfigutation(object):
    def __init__(self, **kwargs):
        self.general_distill = kwargs.pop("general_distill", False)
        self.mlm = kwargs.pop("mlm", False)
        self.clm = kwargs.pop("clm", False)

        #parameters for mlm / clm_poolder layer
        #(bs, seq_len, vocab_size)
        # loss of s - t kl distance
        self.alpha_ce = kwargs.pop("alpha_ce", 0.0 )#if this is 0, disable this loss
        # loss of s - label cross loss of mlm
        self.alpha_mlm = kwargs.pop("alpha_mlm", 0.)
        # loss of s - label cross loss of clm
        self.alpha_clm = kwargs.pop("alpha_clm", 0.)
        # loss of s-t mse loss
        self.alpha_mse = kwargs.pop("alpha_mse", 0.0)
        # loss of s-t cosine loss
        self.alpha_cos = kwargs.pop("alpha_cos", 0.0)

        #paramters for hidden states
        #(bs, seq_len. hidden_size)

        #match func for hidden_states
        self.use_hidden_states = kwargs.pop("use_hidden_states", False)
        self.hidden_match_type = kwargs.pop("hidden_match_type", 'linear')
        assert self.hidden_match_type in ['linear', 'top', 'bottom', 'only_top', 'only_bottom']
        self.hidden_mse = kwargs.pop("hidden_mse", 0.0)
        self.hidden_cos = kwargs.pop("hidden_cos", 0.0)

        # match func for attentions
        self.use_attentions = kwargs.pop("use_attentions", False)
        self.att_match_type = kwargs.pop("att_match_type", 'linear')
        assert self.att_match_type in ['linear', 'top', 'bottom', 'only_top', 'only_bottom']
        self.att_mse = kwargs.pop("att_mse", 0.0)
        self.att_cos = kwargs.pop("att_cos", 0.0)

        self.task_distill = kwargs.pop("task_distill", False)
        #parameters for task-specific layer
        self.task_ce = kwargs.pop("task_ce", 0.0)
        self.task_cos = kwargs.pop("task_cos", 0.0)
        self.task_cros = kwargs.pop("task_cros", 0.0)
        self.task_pkd = kwargs.pop("task_pkd", 0.0)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, 'model.distiller.config.json')

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_json_file(cls, json_file: str) -> "DistillConfigutation":

        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


if __name__ == "__main__":
    config = DistillConfigutation.from_json_file('models_config/DistillerBertDistill.json')
    os.makedirs('tmp/test')
    config.save_pretrained('tmp/test')
    print('finished')