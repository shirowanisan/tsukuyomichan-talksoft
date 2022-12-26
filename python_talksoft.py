from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

from utils import load_model


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

    def generate_voice(self, text):
        model= self.acoustic_model(text)
        if self.config.use_vocoder_stats_flag:
            mel = self.config.scaler.transform(model[2].cpu())
        else:
            mel = model[1]
        wav = self.vocoder.inference(mel)
        wav = wav.view(-1).cpu().detach().numpy()
        return wav  
