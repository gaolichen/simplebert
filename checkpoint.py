import os
import json
from tensorflow.keras.utils import get_file

class ModelInfo(object):
    def __init__(self, config):
        self.name = config['name']
        self.download_url = config.get('download_url', '')
        self.vocab_file = config.get('vocab_file', 'vocab.txt')
        self.config_file = config['config_file']
        self.checkpoint_file = config['checkpoint_file']
        self.cased = config.get('cased', '0') == '1'
        self.class_name = config['class']
        

class ModuleConfig(object):
    def __init__(self, config_path = None, **kwargs):
        if config_path is None:
            current = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(current, 'simple_bert_config.json')
            
        with open(config_path) as f:
            self._config = json.load(f)

        for key, value in kwargs.items():
            self._config[key] = value

        self.cache_dir = self._config.get('cache_dir', '${HOME_PATH}/.cache')
        self.cache_dir = self._replace_home_path(self.cache_dir)
        
        self._model_infos = dict()

        for c in self._config.get('models', []):
            m = ModelInfo(c)
            self._model_infos[m.name] = m

    def model_info(self, name):
        if name in self._model_infos:
            return self._model_infos[name]
        else:
            raise ValueError(f'Cannot find model {name}.')

    def model_keys(self):
        return list(self._model_infos.keys())

    def get_full_path(self, model_name):
        return os.path.join(self.cache_dir, model_name)

    @staticmethod
    def _replace_home_path(path):
        home_key = '${HOME_PATH}'
        if not home_key in path:
            return path

        if os.name == 'nt':
            # windows
            path = path.replace('/', '\\')
            home_path = os.environ.get('USERPROFILE', '.').strip()
            #home_path = home_path.replace('/', '\\')
            if home_path.endswith('\\'):
                home_path = home_path[:-1]
        else:
            # linux
            path = path.replace('\\', '/')
            home_path = os.environ.get('HOME', '.').strip()
            if home_path.endswith('/'):
                home_path = home_path[:-1]

        return path.replace(home_key, home_path)


class CheckpointManager(object):
    def __init__(self, module_config_or_path):
        super(CheckpointManager, self).__init__()

        if isinstance(module_config_or_path, str):
            self.config = ModuleConfig(module_config_or_path)
        else:
            self.config = module_config_or_path

        self.cache_subdir = 'simple-bert'

    def _get_full_path(self, model_name, *file_name):
        return os.path.join(self.config.cache_dir, self.cache_subdir, model_name, *file_name)
    
    def get_config_path(self, model_name):
        return  self._retrieve_model(model_name, 'config_file')

    def get_vocab_path(self, model_name):
        return  self._retrieve_model(model_name, 'vocab_file')

    def get_checkpoint_path(self, model_name):
        return  self._retrieve_model(model_name, 'checkpoint_file')

    def get_cased(self, model_name):
        info = self.config.model_info(model_name)
        return info.cased

    def get_class(self, model_name):
        info = self.config.model_info(model_name)
        return info.class_name

    def _retrieve_model(self, model_name, attribute):
        info = self.config.model_info(model_name)
        #model_path = self.config.get_full_path(model_name)
        file_path = getattr(info, attribute)
        file_name = self._get_basename(file_path)
        full_path = self._get_full_path(model_name, file_name)
        
        if not os.path.exists(full_path):
            if self._is_http(file_path):
                full_path = self._get_file(fname = file_name,
                                           origin = file_path,
                                      cache_subdir = model_name,
                                      cache_dir = os.path.join(self.config.cache_dir, self.cache_subdir))
            elif len(info.download_url) > 0:
                model_path = self._download_if_not_exist(info.name, info.download_url)
            else:
                raise ValueError(f'Cannot download {file_name}.')
            
        return full_path

    @staticmethod
    def _is_http(path):
        return path.startswith('http://') or path.startswith('https://')
    
    @staticmethod
    def _get_basename(path):
        return path.split('/')[-1]
    
    @staticmethod
    def _get_file(fname, origin, extract = False, cache_subdir = 'datasets', cache_dir = '.'):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        
        return get_file(fname = fname,
                        origin = origin,
                        extract = extract,
                        cache_subdir = cache_subdir,
                        cache_dir = cache_dir)
    
    def _download_if_not_exist(self, module_name, download_url):
        dest_dir = self._get_full_path(module_name)

        if not os.path.exists(dest_dir):        
            filename = download_url.split('/')[-1]
            file_path = self._get_file(fname = filename,
                                 origin = download_url,
                                 extract = True,
                                 cache_subdir = self.cache_subdir,
                                 cache_dir = self.config.cache_dir)
            
            file_name = os.path.basename(file_path).split('.')[0]
            src_dir = os.path.join(self.config.cache_dir, self.cache_subdir, file_name)
            
            if os.path.exists(src_dir):
                os.rename(src_dir, dest_dir)
            
        return dest_dir


module_config = ModuleConfig()
checkpoint_manager = CheckpointManager(module_config)
        
