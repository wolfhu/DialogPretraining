

from .distiller_config import DistilConfig
from .models_configuration import DistillConfigutation


DISTILLER_MODEL_CONFIGURATION = {
    'DistillerBert' : DistillConfigutation.from_json_file('config/models_config/DistillerBertDistill.json'),
    'TinyBertGeneralDistill' : DistillConfigutation.from_json_file('config/models_config/TinyBertGeneralDistill.json'),
    'TinyBertTaskDistillStep1' : DistillConfigutation.from_json_file('config/models_config/TinyBertTaskDistill_step1.json'),
    'TinyBertTaskDistillStep2' : DistillConfigutation.from_json_file('config/models_config/TinyBertTaskDistill_step2.json'),
    'PKD-BertDistill' : DistillConfigutation.from_json_file('config/models_config/PKDBertGeneralDistill.json'),
    'TAKD-BertDistill' : DistillConfigutation.from_json_file('config/models_config/DistillerBertDistill.json'),
    'LSTMTaskDistill' : DistillConfigutation.from_json_file('config/models_config/LSTMTaskDistill.json'),
    'ELMOTaskDistill' : DistillConfigutation.from_json_file('config/models_config/ELMOTaskDistill.json'),
}