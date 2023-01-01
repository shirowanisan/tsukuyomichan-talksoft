import numpy as np
import onnxruntime
from espnet_onnx.tts.tts_model import Text2Speech

from tts_config import TTSConfig
import torch

class ParallelWaveGANGenerator:
    def __init__(self,
                 aux_context_window=2,
                 path=None
                 ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            aux_context_window (int): Context window size for auxiliary feature.
        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.aux_context_window = aux_context_window
        self.session = onnxruntime.InferenceSession(path)

    def inference(self, c=None):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        x = torch.randn(1, 1, len(c) * 300).numpy()
        c = torch.tensor(c, dtype=torch.float)
        c = c.transpose(1, 0).unsqueeze(0)
        c = torch.nn.ReplicationPad1d(self.aux_context_window)(c).numpy()
        ort_inputs = {self.session.get_inputs()[0].name:
                      x, self.session.get_inputs()[1].name: c}
        return self.session.run(None, ort_inputs)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [
            (prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v


class TsukuyomichanTalksoft:
    def __init__(self, model_version='v.1.2.0'):
        self.model_version = model_version
        self.config: TTSConfig = TTSConfig.get_config_from_version(
            model_version)
        self.acoustic_model: Text2Speech = self.get_acoustic_model()
        self.vocoder: ParallelWaveGANGenerator = self.get_vocoder()

    def get_acoustic_model(self):
        acoustic_model = Text2Speech(
            model_dir=f"onnx_models\TSUKUYOMICHAN_MODEL_{self.model_version}"
        )
        acoustic_model.spc2wav = None
        return acoustic_model

    def get_vocoder(self):
        vocoder = ParallelWaveGANGenerator(
            path=self.config.optimized_onnx_vocoder_model_path)
        return vocoder

    def generate_voice(self, text, seed=0):
        np.random.seed(seed)
        mel = self.acoustic_model(text)["feat_gen"]
        wav = self.vocoder.inference(mel)
        #wav = self.vocoder.inference(mel).squeeze(0).transpose(1, 0).view(-1).cpu().detach().numpy()
        return wav[0]
