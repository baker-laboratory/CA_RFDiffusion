'''
Functions for checking that the hydra conf object uses supported options
and that sets of options are compatible with each other.
'''

from omegaconf import OmegaConf, open_dict


obsolete_options_dict = {
    'score_model.weights_path': 'Please use inference.ckpt_path instead.',
    'extra_t1d': 'Please use extra_tXd.',
    'extra_t1d_params': 'Please use extra_tXd_params.',
}

# If an option gets renamed, add it here
# These are only applied to keys that come from within a checkpoint
obsolete_options_translate = {
    'extra_t1d': ['extra_tXd'],
    'extra_t1d_params': ['extra_tXd_params'],
}

def flatten_dictionary(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dictionary(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def alert_obsolete_options(conf):
    '''
    Raises an error if the conf uses any obsolete options.
    '''
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = flatten_dictionary(conf)
    conf = expand_dictionary_layers(conf)

    error_msg = []
    used_obsolete_options = set(conf) & set(obsolete_options_dict)
    for opt in used_obsolete_options:
        opt_note = obsolete_options_dict[opt]
        error_msg.append(f'The option "{opt}" is no longer supported. {opt_note}')

    if error_msg:
        raise ValueError('\n'.join(error_msg))



def delete_key(conf, key_parts):
    '''
    Delete a key from an omega_conf config

    Args:
        conf (omegaconf.dictconfig.DictConfig): The config
        key_parts (list[str]): The key_to_delete.split('.') like ['score_model', 'weights_path']

    Returns:
        conf (omegaconf.dictconfig.DictConfig): The config without that key
    '''
    part = key_parts[0]
    if len(key_parts) == 1:
        del conf[part]
    else:
        delete_key(conf[part], key_parts[1:])

def add_key(conf, key_parts, to_store):
    '''
    Add a key to an omega_conf config

    Args:
        conf (omegaconf.dictconfig.DictConfig): The config
        key_parts (list[str]): The key_to_delete.split('.') like ['score_model', 'weights_path']
        to_store (any): The value to store for this key

    Returns:
        conf (omegaconf.dictconfig.DictConfig): The config with that new key
    '''
    part = key_parts[0]
    if len(key_parts) == 1:
        conf[part] = to_store
    else:
        assert hasattr(conf, part), 'Unclear how to add new categories to conf (like "middle" in conf.middle.field). This is fixable'
        add_key(conf[part], key_parts[1:], to_store)

def get_key(conf, key_parts):
    '''
    Helper function to lookup the value in a conf using key_parts.

    Throws if key does not exist.

    Args:
        conf (omegaconf.dictconfig.DictConfig): The config
        key_parts (list[str]): The key_to_delete.split('.') like ['score_model', 'weights_path']

    Returns:
        value (any): The value for that key
    '''
    part = key_parts[0]
    assert hasattr(conf, part), f'Config does not have key: {part}'
    if len(key_parts) == 1:
        return conf[part]
    else:
        return get_key(conf[part], key_parts[1:])

def expand_dictionary_layers(d_conf):
    '''
    For keys that look like this: extra_t1d_params.radius_of_gyration_v2.low
      Also generate: [extra_t1d_params.radius_of_gyration_v2, extra_t1d_params]

    Stores None to new keys as the function that comes next doesn't need the value

    Args:
        d_conf (dict): The omegaconf config as a dictionary

    Returns:
        d_conf (dict): The omegaconf config as a dictionary but expanded
    '''
    for key in list(d_conf):
        if '.' in key:
            sp = key.split('.')
            for i in range(1, len(sp)):
                new_key = '.'.join(sp[:-i])
                d_conf[new_key] = None
    return d_conf

def translate_obsolete_weight_options(conf):
    '''
    Translates options that have been renamed

    Args:
        conf (omegaconf.dictconfig.DictConfig): The config

    Returns:
        conf (omegaconf.dictconfig.DictConfig): The config with keys translated
    '''

    d_conf = OmegaConf.to_container(conf, resolve=True)
    d_conf = flatten_dictionary(d_conf)
    d_conf = expand_dictionary_layers(d_conf)


    needs_rename = set(d_conf) & set(obsolete_options_translate)
    if not needs_rename:
        return conf

    with open_dict(conf):
        for key in needs_rename:
            new_keys = obsolete_options_translate[key]

            print(f'Translating obsolete key: {key} -> {new_keys}')

            value = get_key(conf, key.split('.'))
            delete_key(conf, key.split('.'))

            for new_key in new_keys:
                add_key(conf, new_key.split('.'), value)

    return conf




