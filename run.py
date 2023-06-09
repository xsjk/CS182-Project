import torch
import pytorch_lightning as pl
import soundfile as sf
import os
import argparse
from rich import print
from rich import traceback
from trainer import System

traceback.install()


# # 'from_pretrained' automatically uses the right model class (asteroid.models.DPRNNTasNet).
# model = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")

# # You can pass a NumPy array:
# mixture, _ = sf.read("female-female-mixture.wav", dtype="float32", always_2d=True)
# # Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)
# mixture = mixture.transpose()
# mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])
# out_wavs = model.separate(mixture)

# # Or simply a file name:
# model.separate("female-female-mixture.wav")


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        required=True,
        choices=["DPRNN", "ConvTas"],
        help="The model to use, default is DPRNN.",
    )
    parser.add_argument(
        "--input",
        action="store",
        type=str,
        required=True,
        help="the path for the input wav file",
    )
    parser.add_argument(
        "--output",
        action="store",
        type=str,
        help="the path for the output folder, the default folder is the input file folder",
    )
    args = parser.parse_args()

    config = vars(args)
    if config["model"] == "DPRNN":
        filename = "DPRNNTasNet.ckpt"
        from model import DPRNNTasNet

        config["model"] = DPRNNTasNet(2)
    elif config["model"] == "ConvTas":
        filename = "ConvTasNet.ckpt"
        from model import ConvTasNet

        config["model"] = ConvTasNet(2)
    elif config["model"] == "SWave":
        filename = "SWaveNet.ckpt"
        from model import SWaveNet

        config["model"] = SWaveNet(2)
    else:
        raise RuntimeError("Unknow model")
    checkpoint = torch.load(filename)
    config["system"] = System(config["model"], None, None)
    config["system"].load_state_dict(checkpoint["state_dict"])
    config["model"] = config["system"].model.eval()

    if not os.path.exists(config["input"]):
        raise RuntimeError("The input file doesn't exist")
    if not config["input"].endswith(".wav"):
        raise RuntimeError("The input file must be in WAV format")

    if config["output"] == None:
        config["output"] = os.path.abspath(os.path.dirname(config["input"]))

    # print(config)
    return config


if __name__ == "__main__":
    config = parse_args()

    mixture, samplerate = sf.read(config["input"], dtype="float32", always_2d=True)
    with torch.no_grad():
        out_wavs = config["model"].separate(mixture.T)
    filenameNoExtension = os.path.splitext(config["input"])[0]
    sf.write(f"{filenameNoExtension}-0.wav", out_wavs[0, 0, :], samplerate=samplerate)
    sf.write(f"{filenameNoExtension}-1.wav", out_wavs[0, 1, :], samplerate=samplerate)
