import numpy as np
import onnxruntime
from espnet_onnx.tts.tts_model import Text2Speech

from tts_config import TTSConfig

#ONNXで推論するためのクラスです

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
    def inference(self, c:np.ndarray):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        def replicate_padding(arr):
            """Perform replicate padding on a numpy array."""
            new_pad_shape = (arr[0].shape[0]-4,arr[0].shape[1]) # 2 indicates the width + height to change, a (512, 512) image --> (514, 514) padded image.
            padded_array = np.zeros(new_pad_shape) #create an array of zeros with new dimensions
            arr_shape =(1,arr[0].shape[0]-4,arr[0].shape[1])
            # perform replications
            tmp_arr = arr[0]             
            padded_array = tmp_arr[2:-2]        # result will be zero-pad
            padded_array[0] = tmp_arr[0]            # perform edge pad for top row
            padded_array[-1] = tmp_arr[-1]     # edge pad for bottom row
            padded_array.T[0] = tmp_arr.T[0,2:-2]   # edge pad for first column
            padded_array.T[-1] = tmp_arr.T[-1,2:-2] # edge pad for last column
            
            #at this point, all values except for the 4 corners should have been replicated
            padded_array[0][0] = tmp_arr[0][0]     # top left corner
            padded_array[-1][0] = tmp_arr[-1][0]   # bottom left corner
            padded_array[0][-1] = tmp_arr[0][-1]   # top right corner 
            padded_array[-1][-1] = tmp_arr[-1][-1] # bottom right corner

            result = np.zeros(arr_shape)
            result[0] = padded_array
            return result
        x = np.random.randn(1, 1, len(c) * 300).astype(np.float32) #torch.randn
        c = np.expand_dims(c.transpose(1,0),0) #transposeは
        c = np.pad(c,2,"edge")
        c = replicate_padding(c).astype(np.float32)
        ort_inputs = {self.session.get_inputs()[0].name:
                      x, self.session.get_inputs()[1].name: c}
        
        return self.session.run(None, ort_inputs)


class TsukuyomichanTalksoft:
    def __init__(self, model_version='v.1.2.0'):
        self.model_version = model_version
        self.config: TTSConfig = TTSConfig.get_config_from_version(
            model_version)
        self.acoustic_model: Text2Speech = self.get_acoustic_model()
        self.vocoder: ParallelWaveGANGenerator = self.get_vocoder()
    def get_acoustic_model(self):
        acoustic_model = Text2Speech(
            model_dir=f"onnx_models\TSUKUYOMICHAN_MODEL_{self.model_version}",
            use_quantized = True
        )
        acoustic_model.spc2wav = None
        return acoustic_model

    def get_vocoder(self):
        vocoder = ParallelWaveGANGenerator(
            path=self.config.quant_onnx_vocoder_model_path)
        return vocoder
    def generate_voice(self, text, seed=0):
        np.random.seed(seed)
        mel = self.acoustic_model(text)["feat_gen"]
        wav = self.vocoder.inference(mel)
        #wav = self.vocoder.inference(mel).squeeze(0).transpose(1, 0).view(-1).cpu().detach().numpy()
        return wav[0]
