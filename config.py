import os
import json
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(BASE_DIR, './configs/default.json')


class NameSpace(object):
    #  def __setattr__(self, key, value):
    #  raise AttributeError('Please don\'t modify config dict')

    def __repr__(self):
        return 'config:\n' + self.repr(4)[:-1]  # No newline at end

    def reset(self):
        self.__dict__ = dict()

    def repr(self, indent):
        s = ''
        for k, v in self.__dict__.items():
            if isinstance(v, NameSpace):
                s += '%s%s:\n%s' % (' ' * indent, k, v.repr(indent + 4))
            else:
                s += '%s%s: %s\n' % (' ' * indent, k, v)
        return s

    def has(self, key):
        return key in self.__dict__.keys()


def dump_to_namespace(ns, d):
    for k, v in d.items():
        if isinstance(v, dict):
            if k not in ns.__dict__.keys():
                leaf_ns = NameSpace()
                ns.__dict__[k] = leaf_ns
            dump_to_namespace(ns.__dict__[k], v)
        else:
            ns.__dict__[k] = v


def namespace_to_dict(ns, d):
    for k, v in ns.__dict__.items():
        if isinstance(v, NameSpace):
            d[k] = dict()
            namespace_to_dict(v, d[k])
        else:
            d[k] = v


configGlobal = NameSpace()


def reset_config():
    global configGlobal
    # TODO: Make this work correctly (really reset attributes)
    configGlobal.reset()
    with open(default_config_path, 'r') as handle:
        dump_to_namespace(configGlobal, json.load(handle))


reset_config()


def load_config(filename):
    global globalConfig
    assert filename.endswith('.json')
    name = os.path.basename(filename)[:-5]
    with open(filename, 'r') as handle:
        dump_to_namespace(configGlobal, json.load(handle))
    configGlobal.__dict__["name"] = name
    configGlobal.data.__dict__["basename"] = os.path.basename(configGlobal.data.basepath)
    configGlobal.logging.__dict__["logdir"] = configGlobal.logging.basedir + f'/{name}'
    if configGlobal.evaluation.has('special'):
        if configGlobal.evaluation.special.mode == 'icp':
            configGlobal.logging.__dict__["logdir"] = configGlobal.logging.basedir + f'/icp_{configGlobal.data.basename}/{name}'

    TRAIN_INDICES = provider.getDataFiles(f'{configGlobal.data.basepath}/split/train.txt')
    VAL_INDICES = provider.getDataFiles(f'{configGlobal.data.basepath}/split/val.txt')
    configGlobal.data.__dict__["ntrain"] = len(TRAIN_INDICES)
    configGlobal.data.__dict__["nval"] = len(VAL_INDICES)


def save_config(filename):
    global globalConfig
    assert filename.endswith('.json')
    with open(filename, 'w') as handle:
        d = dict()
        namespace_to_dict(configGlobal, d)
        json.dump(d, handle)
