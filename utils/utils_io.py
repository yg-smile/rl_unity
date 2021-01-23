import json
import pickle
import os
from collections import namedtuple


def _save_config(config={},
                 dir_name='res',
                 file_name="unnamed_file"):
    if type(config) is not dict:
        raise TypeError('save_config expects dict')
    file_name += '.txt'
    dir_name = './' + dir_name
    dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    json_name = file_name
    with open(dir_name + json_name, 'w') as file:
        file.write(json.dumps(config, indent=2))  # use `json.loads` to do the reverse


def _load_config(dir_name='res',
                 file_name="unnamed_file"):
    file_name += '.txt'
    dir_name = './' + dir_name
    dir_name += '/'

    json_name = file_name
    with open(dir_name + json_name, 'r') as file:
        config = json.loads(file.read())
    return config


def _save_res(res={},
              file_name="unnamed_file",
              dir_name="res"):
    if type(res) is not dict:
        raise TypeError('save_res expects dict')
    file_name += '.pkl'
    dir_name = './' + dir_name
    dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pkl_name = file_name
    with open(dir_name + pkl_name, 'wb') as file:
        pickle.dump(res, file)


def _load_res(dir_name='res',
              file_name="unnamed_file"):
    file_name += '.pkl'
    dir_name = './' + dir_name
    dir_name += '/'

    pkl_name = file_name
    with open(dir_name + pkl_name, 'rb') as file:
        res = pickle.load(file)
    return res


def _file_dir_name_convention(config):
    # generate naming convention of io file/dir
    file_name = "{}_{}_seed{}".format(config['env'], config['algo'], config['seed'])
    dir_name = "res"
    ext_name = "_ext"+config['save_file_ext']
    convention = namedtuple("convention", ["file_name", "dir_name", "ext_name"])
    return convention(file_name=file_name, dir_name=dir_name, ext_name=ext_name)


def save_run(config,
             res,
             file_name=None,
             dir_name=None):
    '''
    :param config: dict contains the meta info about the simulation along with all important hyperparameters
    :param res: dict contains the simulation deliverables
    :param file_name: optional file name specify the saved file name
    :param dir_name: optional directory name specify the saved dir name
    :return:
    '''
    if file_name is None:
        convention = _file_dir_name_convention(config)
        file_name = convention.file_name + convention.ext_name
    if dir_name is None:
        convention = _file_dir_name_convention(config)
        dir_name = convention.dir_name
    _save_config(config=config,
                 file_name=file_name,
                 dir_name=dir_name)
    _save_res(res=res,
              file_name=file_name,
              dir_name=dir_name)


def load_run(config,
             file_name=None,
             dir_name=None):
    '''
    :param config: dict contains the meta info about the simulation along with all important hyperparameters
    :param file_name: optional file name specify the saved file name
    :param dir_name: optional directory name specify the saved dir name
    :return:
    '''
    if file_name is None:
        convention = _file_dir_name_convention(config)
        file_name = convention.file_name + convention.ext_name
    if dir_name is None:
        convention = _file_dir_name_convention(config)
        dir_name = convention.dir_name
    config_load = _load_config(file_name=file_name,
                               dir_name=dir_name)
    res_load = _load_res(file_name=file_name,
                         dir_name=dir_name)
    return config_load, res_load
