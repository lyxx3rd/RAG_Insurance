import os
import yaml

def load_config(type):
    abs_path = os.path.dirname(__file__)
    if type == "QA":
        CONFIG_PATH = os.path.join(abs_path, 'QA_config.yaml')
    elif type == "Contract":
        CONFIG_PATH = os.path.join(abs_path, 'Contract_config.yaml')
    else:
        print("config type error")
    app_conf = {}
    try:
        app_conf = yaml.load(open(CONFIG_PATH, 'r', encoding='UTF-8').read(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        app_conf = {}
        print('config.yaml not exist !')
    LOG_ROOT = app_conf['log_dir']
    return app_conf