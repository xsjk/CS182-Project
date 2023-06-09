import torch
from torch import optim
from loss import PILPairwiseSDR
from trainer import System, Trainer
from argparse import ArgumentParser
from rich import print
from rich import traceback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_lightning.strategies import ddp

traceback.install()


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, required=True,
                        choices=[
                            "DPRNN", 
                            "ConvTas",
                            "SWave"
                        ],
                        help="The model to train")
    parser.add_argument("--optimizer", action="store", type=str, default="Adam",
                        choices=[
                            "Adam"
                        ],
                        help="The optimizer for training")
    parser.add_argument("--batch_size", action="store", type=int, default=16,
                        help="The batch size of the dataloader")
    parser.add_argument("--n_src", action="store", type=int, default=2,
                        help="The number of channel to separate")
    parser.add_argument("--n_episode", action="store", type=int, default=100,
                        help="The number of episode to train")
    parser.add_argument("--resume", action="store", type=str, metavar="CHECKPOINT_PATH",
                        help="The .ckpt file from which to resume")
    parser.add_argument("--logger_enabled", action="store_true", default=True,
                        help="Whether to activate tensorboard logger")
    parser.add_argument("--logger_path", action="store", type=str,
                        help="The path for the tensorboard logger")


    args = parser.parse_args()

    config = vars(args)

    loss = PILPairwiseSDR

    if config["model"] == "DPRNN":
        # from asteroid.models import DPRNNTasNet
        from model import DPRNNTasNet
        config["model"] = DPRNNTasNet(n_src=config["n_src"])
        
    elif config["model"] == "ConvTas":
        # from asteroid.models import ConvTasNet
        from model import ConvTasNet
        config["model"] = ConvTasNet(n_src=config["n_src"])
        pass
    elif config["model"] == "SWave":
        from model import SWaveNet
        config["model"] = SWaveNet(n_src=config["n_src"])
    else:
        pass

    if config["logger_enabled"]:
        if config["logger_path"] == None:
            config["logger_path"] = "runs/"+config["model"].__class__.__name__
        config["logger"] = TensorBoardLogger(config["logger_path"])
    else:
        config["logger"] = None

    if config["optimizer"] == "Adam":
        config["optimizer"] = optim.Adam(config["model"].parameters(), lr=1e-3)

    from data import LibriMix
    config["dataset"] = LibriMix.loaders_from_mini(task="sep_clean", batch_size=config["batch_size"])


    config["system"] = System(config["model"], config["optimizer"], loss)
    # config["trainer"] = Trainer(max_epochs=config["n_episode"], logger=config["logger"], log_every_n_steps=1)
    config["trainer"] = Trainer(max_epochs=config["n_episode"], logger=config["logger"], log_every_n_steps=1, strategy='ddp_find_unused_parameters_true')

    if config["resume"] != None:
        checkpoint = torch.load(config["resume"])
        config["system"].load_state_dict(checkpoint["state_dict"])
    return config



if __name__ == "__main__":

    config = parse_args()
    train_loader, val_loader = config["dataset"]
    trainer = config["trainer"]
    trainer.fit(config["system"], train_loader, val_loader)
    print(f'write to {config["model"].__class__.__name__}.ckpt')
    trainer.save_checkpoint(config["model"].__class__.__name__+".ckpt")

