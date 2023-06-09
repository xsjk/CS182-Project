import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model import DPRNNTasNet

model = DPRNNTasNet

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='path/to/save/checkpoints',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

trainer = Trainer(
    resume_from_checkpoint='path/to/your/checkpoint.ckpt',
    callbacks=[checkpoint_callback]
)