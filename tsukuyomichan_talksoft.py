print("initializing talksoft on Background")
import time
init_time = time.perf_counter()
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

from tts_config import TTSConfig

class TsukuyomichanTalksoft:
    def __init__(self, model_version='v.1.2.0'):
        self.config: TTSConfig = TTSConfig.get_config_from_version(model_version)
        self.acoustic_model = self.get_acoustic_model()
        self.vocoder = self.get_vocoder()
    
    def get_acoustic_model(self):
        acoustic_model = Text2Speech(
            self.config.acoustic_model_config_path,
            self.config.acoustic_model_path,
            device=self.config.device,
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
        vocoder = load_model(self.config.vocoder_model_path).to(self.config.device).eval()
        vocoder.remove_weight_norm()
        return vocoder

    def generate_voice(self, text, seed):
        generate_voice = time.perf_counter()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        acoustic_time = time.perf_counter()
        _, mel, mel_dnorm, *_ = self.acoustic_model(text)
        print("acoustic_time:"+str(time.perf_counter()- acoustic_time))
        if self.config.use_vocoder_stats_flag:
            print("Using Vocoder...")
            mel = self.config.scaler.transform(mel_dnorm.cpu())
        vocoder_time = time.perf_counter()
        wav = self.vocoder.inference(mel)
        print("vocoder_time:"+str(time.perf_counter()- vocoder_time))
        wav_time = time.perf_counter()
        wav = wav.view(-1).cpu().detach().numpy()
        print("wav_time:"+str(time.perf_counter()- wav_time))
        print("generate_voice:"+str(time.perf_counter()- generate_voice))
        return wav  
print("talksoft init time:"+str(time.perf_counter()- init_time))