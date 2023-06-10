import torch
import soundfile as sf
import os
import argparse
from rich import traceback
from trainer import System
traceback.install()

def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        required=True,
        choices=["DPRNN", "ConvTas", "SWave"],
        help="The model to use.",
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
    
    if not os.path.exists(filename):
        raise RuntimeError("You have to train the model first")
    
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

    return config


if __name__ == "__main__":
    config = parse_args()

    mixture, samplerate = sf.read(config["input"], dtype="float32", always_2d=True)
    with torch.no_grad():
        out_wavs = config["model"].separate(mixture.T)
    filenameNoExtension = os.path.splitext(config["input"])[0]
    sf.write(f"{filenameNoExtension}-0.wav", out_wavs[0, 0, :], samplerate=samplerate)
    sf.write(f"{filenameNoExtension}-1.wav", out_wavs[0, 1, :], samplerate=samplerate)
