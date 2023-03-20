import os
import yaml
from typing import NamedTuple


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
    onnx_model_path: str
    onnx_vocoder_model_path: str
    optimized_onnx_vocoder_model_path: str
    quant_onnx_vocoder_model_path: str
    use_vocoder_stats_flag: bool
    device: str

    @classmethod
    def get_config_from_version(cls, model_version: str, download_path: str = './models', onnx_path: str = "./onnx_models"):
        import torch
        if model_version == 'v.1.0.0':
            model_url = 'https://drive.google.com/uc?id=1fuI0WrISJt5Gf9rNepSJFIAlupeeC8_V'
            acoustic_name = '200epoch.pth'
            onnx_vocoder_name = 'ParallelWaveGANGenerator.onnx'
            optimized_onnx_vocoder_name = 'ParallelWaveGANGenerator.opt.onnx'
            quant_onnx_vocoder_name  = 'ParallelWaveGANGenerator.quant.onnx'
            vocoder_name = 'checkpoint-400000steps.pkl'
            use_vocoder_stats_flag = False
        elif model_version == 'v.1.1.0':
            model_url = 'https://drive.google.com/uc?id=1FyDR366PvdWejWI0WJ9rNaCAEiiLewPv'
            acoustic_name = '200epoch.pth'
            onnx_vocoder_name = 'ParallelWaveGANGenerator.onnx'
            optimized_onnx_vocoder_name = 'ParallelWaveGANGenerator.opt.onnx'
            quant_onnx_vocoder_name = 'ParallelWaveGANGenerator.quant.onnx'
            vocoder_name = 'checkpoint-300000steps.pkl'
            use_vocoder_stats_flag = False
        elif model_version == 'v.1.2.0':
            model_url = 'https://drive.google.com/uc?id=1scfGUohN2QTT4w6XTrKX2FPvm8yuhA1f'
            acoustic_name = '200epoch.pth'
            onnx_vocoder_name = 'ParallelWaveGANGenerator.onnx'
            optimized_onnx_vocoder_name = 'ParallelWaveGANGenerator.opt.onnx'
            quant_onnx_vocoder_name  = 'ParallelWaveGANGenerator.quant.onnx'
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
        onnx_model_path = f"{onnx_path}/TSUKUYOMICHAN_MODEL_{model_version}"
        onnx_vocoder_model_path = f"{onnx_model_path}/vocoder/{onnx_vocoder_name}"
        optimized_onnx_vocoder_model_path = f"{onnx_model_path}/vocoder/{optimized_onnx_vocoder_name}"
        quant_onnx_vocoder_model_path = f"{onnx_model_path}/vocoder/{quant_onnx_vocoder_name}"

        #保存するフォルダがなかった場合はフォルダを作る
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        #利用するpytorchのモデルがなかった場合はモデルをダウンロードする
        if not os.path.exists(model_path):
            cls.download_model(download_path, model_path, model_url)
            cls.update_acoustic_model_config(
                acoustic_model_config_path, acoustic_model_stats_path)
        
        #利用するonnxのモデルがなかった場合はモデルをpytorchから変換し、量子化する
        if not os.path.exists(onnx_model_path):
            from espnet_onnx.export import TTSModelExport
            from espnet2.bin.tts_inference import Text2Speech
            print("TTSModel exporting...")
            m = TTSModelExport(onnx_path)
            acoustic_model = Text2Speech(
                acoustic_model_config_path,
                acoustic_model_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=10.0,
                use_att_constraint=False,
                backward_window=1,
                forward_window=3
            )
            m.export(acoustic_model,f"TSUKUYOMICHAN_MODEL_{model_version}", quantize=True)
            print("exported")
            cls.update_onnx_acoustic_model_config(f"{onnx_model_path}/config.yaml")
        #利用するonnxのボコーダーのモデルのフォルダがなかった場合はフォルダを作成する
        if not os.path.exists(f"{onnx_model_path}/vocoder/"):
            os.makedirs(f"{onnx_model_path}/vocoder/")
        #利用するonnxのボコーダーのモデルのファイルがなかった場合はpytorchから変換し、量子化する
        if not os.path.exists(onnx_vocoder_model_path):
            import torch.onnx
            from parallel_wavegan.models import ParallelWaveGANGenerator
            checkpoint = vocoder_model_path
            config = None
            if config is None:
                dirname = os.path.dirname(checkpoint)
                config = os.path.join(dirname, "config.yml")
                with open(config) as f:
                    config = yaml.load(f, Loader=yaml.Loader)
                # get model and load parameters
                model = ParallelWaveGANGenerator(**config["generator_params"])
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu")[
                        "model"]["generator"]
                )

            def Convert_ONNX():
                # モデルを推論モードにする
                model.eval()
               
                #サンプルの入力を作る
                sample_c = torch.randn(40, 80)
                sample_x = torch.randn(1, 1, 12000)
                sample_c = torch.nn.ReplicationPad1d(2)(
                    sample_c.transpose(1, 0).unsqueeze(0)
                )
                # Export the model
                torch.onnx.export(model,                   # model being run
                                  (sample_x, sample_c),    # model input (or a tuple for multiple inputs)
                                  onnx_vocoder_model_path, # where to save the model
                                  export_params=True,      # store the trained parameter weights inside the model file
                                  do_constant_folding=True,# whether to execute constant folding for optimization
                                  input_names=["x", "c"],  # the model's input names
                                  output_names=["audio"],  # the model's output names
                                  dynamic_axes={"x": {2: "x_seq"},
                                                "c": {2: "c_seq"},
                                                "audio": {2: "audio_seq"}}
                                  )
                print(" ")
                print('torch vocoder model has been converted to ONNX')
            Convert_ONNX()
        if not os.path.exists(optimized_onnx_vocoder_model_path):
            print("Optimizing ONNX model")
            import onnx
            from onnxsim import simplify
            model = onnx.load(onnx_vocoder_model_path)
            model_opt, check = simplify(model)
            if check:
                print("Model optimized successfully")
                onnx.save(model_opt, optimized_onnx_vocoder_model_path)
            else:
                print("Failed to optimize model")
        if not os.path.exists(quant_onnx_vocoder_model_path):
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                optimized_onnx_vocoder_model_path,
                quant_onnx_vocoder_model_path,
                weight_type=QuantType.QUInt8,
            )
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
                         onnx_model_path=onnx_model_path,
                         onnx_vocoder_model_path=onnx_vocoder_model_path,
                         device=device,
                         optimized_onnx_vocoder_model_path=optimized_onnx_vocoder_model_path,
                         quant_onnx_vocoder_model_path=quant_onnx_vocoder_model_path)

    @staticmethod
    def download_model(download_path, model_path, model_url):
        import gdown
        import zipfile
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
    def update_onnx_acoustic_model_config(onnx_acoustic_model_config_path):
        with open(onnx_acoustic_model_config_path) as f:
            yml = yaml.safe_load(f)
        if not yml['normalize']['use_normalize'] == False or not yml['vocoder']['vocoder_type'] == "not_used":
            yml['normalize']['use_normalize'] = False
            yml['vocoder']['vocoder_type'] = "not_used"
            with open(onnx_acoustic_model_config_path, 'w') as f:
                yaml.safe_dump(yml, f)
            print("Update acoustic model yaml.")
