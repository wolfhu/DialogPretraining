
import json
import logging
import os
import copy

logger = logging.getLogger(__name__)

class DistilConfig(object):
    def __init__(self, **kwargs):
        self.model_type = kwargs.pop("model_type", None)
        self.teacher_model = kwargs.pop("teacher_architectures", None)
        self.student_model = kwargs.pop("student_architectures", None)

        self.task_name = kwargs.pop("task_name", None)
        self.n_epoch = kwargs.pop('num_train_epochs', 10)
        self.max_seq_length = kwargs.pop('max_seq_length', 128)
        self.warmup_steps = kwargs.pop('warmup_steps', 0)
        self.warmup_percentage = kwargs.pop('warmup_percentage', 0.01)
        # Batch size per GPU/CPU for training.
        self.per_gpu_train_batch_size = kwargs.pop("per_gpu_train_batch_size", 8)
        self.per_gpu_eval_batch_size = kwargs.pop("per_gpu_eval_batch_size", 8)
        self.learning_rate = kwargs.pop("learning_rate", 5e-5)

        #data
        self.data_dir = kwargs.pop("data_dir", None)
        self.output_dir = kwargs.pop("output_dir", None)

        #train or evaluate
        self.do_train = kwargs.pop("do_train", False)
        self.do_eval = kwargs.pop("do_eval", False)
        self.do_lower_case = kwargs.pop("do_lower_case", True)
        self.output_mode = kwargs.pop("output_mode", None)

        assert self.do_train or self.do_eval, "do_train and do_eval cannot be false at the same time"

        #distill model
        self.theseus_scheduler = kwargs.pop('theseus_scheduler', None)
        self.dc_bert_model_config = kwargs.pop('dc_bert_model_config', None)
        self.rnn_config = kwargs.pop('rnn_config', None)
        self.general_distiller_scheduler = kwargs.pop('general_distiller_scheduler', None)
        self.tiny_bert_scheduler = kwargs.pop('tiny_bert_scheduler', None)


        # Number of updates steps to accumulate before performing a backward/update pass.

        #fp16
        self.fp16 = kwargs.pop("fp16", False)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", False)

        self.local_rank = kwargs.pop("local_rank", -1)

        self.distiller_config_check()

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, 'distiller.config.json')

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    def distiller_config_check(self):
        # assert self.model_type in ['dc-bert', 'theseus-bert', 'tiny-bert', '']
        if self.model_type == 'dc-bert':
            assert self.dc_bert_model_config is not None, "dc_bert_model_config cannot be none for dc-bert"
        if self.model_type == 'theseus-bert':
            assert self.theseus_scheduler is not None, "theseus_scheduler cannot be none for theseus-bert"
        if self.model_type == 'rnn':
            assert self.rnn_config is not None, "rnn_config cannot be none for rnn"
        if self.model_type == 'tiny-bert':
            assert self.tiny_bert_scheduler is not None, "tiny_bert_scheduler cannot be none for tiny-bert"

        assert self.teacher_model is not None,  "Config for teacher model or model"
        assert self.teacher_model is not None and self.teacher_model['model_type'] is not None \
               and self.teacher_model['config_file'] is not None and self.teacher_model['model_name_or_path'] is not None

        if 'distiller' in self.model_type:
            # if model type is distiller in model_type, there should be a student model
            # for dc-bert and theseus bert, student model is not needed
            assert self.student_model is not None and self.student_model[
                'model_type'] is not None \
                   and self.student_model['config_file'] is not None

        if not os.path.exists(self.data_dir):
            logger.warning("data directory () must exist and  not empty.")
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            logger.warning("Output directory () already exists and is not empty.")

    @classmethod
    def from_json_file(cls, json_file: str) -> "DistilConfig":

        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str) :
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
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

if __name__ == '__main__':
    config = DistilConfig.from_json_file('distill.example.1.json')
    print('finished')