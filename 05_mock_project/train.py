import torch
from torch import nn
import wandb
import einops

from ai_image_models.data import PixelArtDataset
from ai_image_models.models import FlowMLP
from ai_image_models.learner import Learner, train_classifier
from ai_image_models.eval import fid

class FlowCNN(nn.Module):
    def __init__(self, img_shape=(16, 16, 3), h=128):
        super().__init__()
        self.img_shape = img_shape

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(in_channels=h, out_channels=h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(in_channels=h, out_channels=h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(in_channels=h, out_channels=3, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        )

    def forward(self, z, t):
        z = einops.rearrange(z, 'b h w c -> b c h w')
        z = self.net(z)
        return einops.rearrange(z, 'b c h w -> b h w c')



def evaluate(learner, clf, ds):
  real = clf.features(ds.x[:2048])
  gen = clf.features(learner.generate(n=2048, steps=100))
  return fid(real, gen)


def main():
  config = {
   'lr': 1e-4,
   'epochs': 1,
   'batch_size': 256,
   'arch': 'cnn'
  }
  run = wandb.init(project="ai_image_model_04", config=config)
  cfg = run.config

  ds = PixelArtDataset()
  if cfg.arch == 'mlp':
    model = FlowMLP(img_shape=(16, 16, 3))
  elif cfg.arch == 'cnn':
    model = FlowCNN(img_shape=(16, 16, 3))

  learner = Learner(model, lr=cfg.lr)
  learner.learn(ds, epochs=cfg.epochs, batch_size=cfg.batch_size, logger=run)

  clf = train_classifier(ds)
  run.summary['FID'] = evaluate(learner, clf, ds)


if __name__ == "__main__":
  main()
