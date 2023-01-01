import onnxruntime
from espnet_onnx.tts.tts_model import Text2Speech
from tts_config import TTSConfig
import torch
import numpy as np
from parallel_wavegan import models
from parallel_wavegan.layers import upsample
import module
class ParallelWaveGANGenerator:
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=30,
                 stacks=3,
                 aux_channels=80,
                 aux_context_window=2,
                 use_causal_conv=False,
                 upsample_conditional_features=True,
                 upsample_net="ConvInUpsampleNetwork",
                 upsample_params={"upsample_scales": [4, 4, 4, 4]},
                 path=None
                 ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.session = onnxruntime.InferenceSession(path)
        
         # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define conv + upsampling network
        if upsample_conditional_features:
            upsample_params.update({
                "use_causal_conv": use_causal_conv,
            })
            if upsample_net == "MelGANGenerator":
                assert aux_context_window == 0
                upsample_params.update({
                    "use_weight_norm": False,  # not to apply twice
                    "use_final_nonlinear_activation": False,
                })
                self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update({
                        "aux_channels": aux_channels,
                        "aux_context_window": aux_context_window,
                    })
                self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1
    def inference(self, c=None):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).

        Returns:
            Tensor: Output tensor (T, out_channels)

        """    
        x = torch.randn(1, 1, len(c) * 300)
        c = torch.tensor(c, dtype=torch.float)
        c = c.transpose(1, 0).unsqueeze(0)
        c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        def to_numpy(tensor):
            return tensor.numpy()
        ort_inputs = {self.session.get_inputs()[0].name: to_numpy(x),self.session.get_inputs()[1].name:to_numpy(c) }
        return self.session.run(None, ort_inputs)
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
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
        self.config: TTSConfig = TTSConfig.get_config_from_version(model_version)
        self.acoustic_model = self.get_acoustic_model()
        self.vocoder = self.get_vocoder()
    
    def get_acoustic_model(self):
        acoustic_model = Text2Speech(
            model_dir=f"onnx_models\TSUKUYOMICHAN_MODEL_{self.model_version}"
        )
        acoustic_model.spc2wav = None
        return acoustic_model

    def get_vocoder(self):
        vocoder = ParallelWaveGANGenerator(path=self.config.onnx_vocoder_model_path)
        return vocoder
    def generate_voice(self, text,seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            mel = self.acoustic_model(text)["feat_gen"]
        wav = self.vocoder.inference(mel)
        #wav = self.vocoder.inference(mel).squeeze(0).transpose(1, 0).view(-1).cpu().detach().numpy()
        return wav[0]


    