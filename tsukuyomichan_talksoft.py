import os
import zipfile

import gdown
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model


class TsukuyomichanTalksoft:
    def __init__(self, model_version='v.1.1.0'):
        if model_version == 'v.1.0.0':
            self.model_url = 'https://drive.google.com/uc?id=1sxX2D1ioNXmo8QLScufnkqpgpv1NnGka'
            self.folder_name = 'TSUKUYOMICHAN_MODEL'
            self.vocoder_model_path = f"{self.folder_name}/VOCODER/checkpoint-400000steps.pkl"
        elif model_version == 'v.1.1.0':
            self.model_url = 'https://drive.google.com/uc?id=1FyDR366PvdWejWI0WJ9rNaCAEiiLewPv'
            self.folder_name = 'TSUKUYOMICHAN_MODEL_v.1.1.0'
            self.vocoder_model_path = f"{self.folder_name}/VOCODER/checkpoint-300000steps.pkl"
        else:
            raise Exception("存在しないモデルバージョンです")

        self.acoustic_model_path = f"{self.folder_name}/ACOUSTIC_MODEL/200epoch.pth"
        self.acoustic_model_config_path = f"{self.folder_name}/ACOUSTIC_MODEL/config.yaml"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not os.path.exists(self.folder_name):
            self.download_model()
        self.acoustic_model = self.get_acoustic_model()
        self.vocoder = self.get_vocoder()

    def download_model(self):
        zip_name = f"{self.folder_name}.zip"
        gdown.download(self.model_url, zip_name, quiet=True)
        with zipfile.ZipFile(zip_name) as model_zip:
            model_zip.extractall('')
        os.remove(zip_name)

    def get_acoustic_model(self):
        acoustic_model = Text2Speech(
            self.acoustic_model_config_path,
            self.acoustic_model_path,
            device=self.device,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3
        )
        acoustic_model.spc2wav = None
        return acoustic_model

    def get_vocoder(self):
        vocoder = load_model(self.vocoder_model_path).to(self.device).eval()
        vocoder.remove_weight_norm()
        return vocoder

    def generate_voice(self, text, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            _, mel, *_ = self.acoustic_model(text)
            wav = self.vocoder.inference(mel)
        wav = wav.view(-1).cpu().numpy()
        return wav
