import yaml
import pprint

class Dict2Attr(object):
    def __init__(self, d):
        self.__dict__ = d
        self.original_dict = d
    def print(self):
        pprint.pprint(self.original_dict)
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return output_dict

def preprocess_config(config):
    """
    Preprocessing yaml config if needed.
    :param config: Raw config.
    :return: processed config.
    """
    # print('config params')
    keys = set()
    for key in config:
        # print(key, config[key])
        keys.add(key)
        if type(config[key]) == str:
            if config[key].startswith('$data_root'):
                config[key] = config['data_root'] + config[key][len('$data_root'):]
                # print(key, config[key])
            elif config[key].startswith('$model_root'):
                config[key] = config['model_root'] + config[key][len('$model_root'):]
                # print(key, config[key])
            elif config[key].startswith('$'):
                raise NotImplementedError
    print('config params')
    if 'data_augment' not in keys:
        config['data_augment'] = False
    if 'LLM_device' not in keys:
        config['LLM_device'] = 0
    if 'finetune_from_ours' not in keys:
        config['finetune_from_ours'] = False
        
    return config

def load_and_apply_yaml_config(yaml_file):
    with open(yaml_file, 'r') as fp:
        config = yaml.full_load(fp)
    config = preprocess_config(config)
    config = Dict2Attr(config)
    return config