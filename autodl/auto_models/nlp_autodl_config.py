import os
import json
from collections import namedtuple
import copy

#
IF_RESET_TFGRAPH_SESS_RUN = False
TF_DATASET_TO_NUMPY_MODE = "graph"  # eager/graph
#

#
autodl_global_config = {
    "meta_solution": {
        "cv_solution": "DeepWisdom",
        "nlp_solution": "upwind_flys",
        "speech_solution": "PASA_NJU",
    },
    "data_space": {
        "domain_dataset": {
            "text": {"if_sample": True, "sample_ratio": 0.5},
            "speech": {"if_sample": True, "sample_ratio": 0.5},
        }
    },
}


class MetaSoluConf(object):
    def __init__(self):
        self.cv_solution = None
        self.nlp_solution = None
        self.speech_solution = None


class DsDomainDatasetConf(object):
    def __init__(self):
        self.if_sample = None
        self.sample_ratio = None


class DsDomainDatasetSets(object):
    def __init__(self):
        self.text = DsDomainDatasetConf()
        self.speech = DsDomainDatasetConf()


class DsConf(object):
    def __init__(self):
        self.domain_dataset = DsDomainDatasetSets()


class AutoDlConf(object):
    def __init__(self):
        self.meta_solution = MetaSoluConf()
        self.data_space = DsConf()


class ConfigParserA(object):
    def _json_object_hook(self, d):
        return namedtuple("X", d.keys())(*d.values())

    def json2obj(self, data):
        return json.loads(data, object_hook=self._json_object_hook)

    def from_type_autodlconf(self, conf_data) -> AutoDlConf:
        return copy.deepcopy(self.json2obj(json.dumps(conf_data)))


autodl_g_conf_repr = json.dumps(autodl_global_config, indent=4)

config_parser_a = ConfigParserA()
AUTODL_G_CONF = config_parser_a.from_type_autodlconf(autodl_global_config)
META_SOLUS = AUTODL_G_CONF.meta_solution
DM_DS_PARAS = AUTODL_G_CONF.data_space.domain_dataset
