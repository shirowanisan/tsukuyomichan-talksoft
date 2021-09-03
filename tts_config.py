import os
import zipfile
from typing import NamedTuple, Optional

import gdown
import torch
import yaml
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler


class TTSConfig(NamedTuple):
    download_path: str
    model_version: str
    model_url: str
    model_path: str
    acoustic_model_path: str
    acoustic_model_config_path: str
    acoustic_model_stats_path: str
    vocoder_model_path: str
    vocoder_stats_path: str
    use_vocoder_stats_flag: bool
    scaler: Optional[StandardScaler]
    device: str

    @classmethod
    def get_config_from_version(cls, model_version: str, download_path: str = './models'):
        if model_version == 'v.1.0.0':
            model_url = 'https://drive.google.com/uc?id=1fuI0WrISJt5Gf9rNepSJFIAlupeeC8_V'
            acoustic_name = '200epoch.pth'
            vocoder_name = 'checkpoint-400000steps.pkl'
            use_vocoder_stats_flag = False
        elif model_version == 'v.1.1.0':
            model_url = 'https://drive.google.com/uc?id=1FyDR366PvdWejWI0WJ9rNaCAEiiLewPv'
            acoustic_name = '200epoch.pth'
            vocoder_name = 'checkpoint-300000steps.pkl'
            use_vocoder_stats_flag = False
        elif model_version == 'v.1.2.0':
            model_url = 'https://drive.google.com/uc?id=1scfGUohN2QTT4w6XTrKX2FPvm8yuhA1f'
            acoustic_name = '200epoch.pth'
            vocoder_name = 'checkpoint-300000steps.pkl'
            use_vocoder_stats_flag = True
        else:
            raise Exception("存在しないモデルバージョンです")
        model_path = f"{download_path}/TSUKUYOMICHAN_MODEL_{model_version}"
        acoustic_model_path = f"{model_path}/ACOUSTIC_MODEL/{acoustic_name}"
        acoustic_model_config_path = f"{model_path}/ACOUSTIC_MODEL/config.yaml"
        acoustic_model_stats_path = f"{model_path}/ACOUSTIC_MODEL/feats_stats.npz"
        vocoder_model_path = f"{model_path}/VOCODER/{vocoder_name}"
        vocoder_stats_path = f"{model_path}/VOCODER/stats.h5"

        if not os.path.exists(download_path):
            os.makedirs(download_path)
        if not os.path.exists(model_path):
            cls.download_model(download_path, model_path, model_url)
            cls.update_acoustic_model_config(acoustic_model_config_path, acoustic_model_stats_path)

        scaler = cls.get_scaler(vocoder_stats_path) if use_vocoder_stats_flag else None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return TTSConfig(download_path=download_path,
                         model_version=model_version,
                         model_url=model_url,
                         model_path=model_path,
                         acoustic_model_path=acoustic_model_path,
                         acoustic_model_config_path=acoustic_model_config_path,
                         acoustic_model_stats_path=acoustic_model_stats_path,
                         vocoder_model_path=vocoder_model_path,
                         vocoder_stats_path=vocoder_stats_path,
                         use_vocoder_stats_flag=use_vocoder_stats_flag,
                         scaler=scaler,
                         device=device)

    @staticmethod
    def download_model(download_path, model_path, model_url):
        zip_path = f"{model_path}.zip"
        gdown.download(model_url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as model_zip:
            model_zip.extractall(download_path)
        os.remove(zip_path)

    @staticmethod
    def update_acoustic_model_config(acoustic_model_config_path, acoustic_model_stats_path):
        with open(acoustic_model_config_path) as f:
            yml = yaml.safe_load(f)
        if not yml['normalize_conf']['stats_file'] == acoustic_model_stats_path:
            yml['normalize_conf']['stats_file'] = acoustic_model_stats_path
            with open(acoustic_model_config_path, 'w') as f:
                yaml.safe_dump(yml, f)
            print("Update acoustic model yaml.")

    @staticmethod
    def get_scaler(vocoder_stats_path: str) -> StandardScaler:
        stats = vocoder_stats_path
        scaler = StandardScaler()
        scaler.mean_ = read_hdf5(stats, "mean")
        scaler.scale_ = read_hdf5(stats, "scale")
        scaler.n_features_in_ = scaler.mean_.shape[0]
        return scaler
