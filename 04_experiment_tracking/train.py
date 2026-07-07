import torch
import wandb

from ai_image_models.data import PixelArtDataset
from ai_image_models.models import FlowMLP
from ai_image_models.learner import Learner, train_classifier


def fid(f1, f2):
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    eig = torch.linalg.eigvals(c1 @ c2).real.clamp(min=0)
    return (((m1 - m2) ** 2).sum() + torch.trace(c1 + c2) - 2 * eig.sqrt().sum()).item()


def evaluate(learner, clf, ds):
  real = clf.features(ds.x[:2048])
  gen = clf.features(learner.generate(n=2048, steps=100))
  return fid(real, gen)


def main():
  config = {
   'lr': 1e-4,
   'epochs': 10,
   'batch_size': 256
  }
  run = wandb.init(project="ai_image_model_04", config=config)
  cfg = run.config

  ds = PixelArtDataset()
  flow = FlowMLP(img_shape=(16, 16, 3))
  learner = Learner(flow, lr=cfg.lr)
  learner.learn(ds, epochs=cfg.epochs, batch_size=cfg.batch_size, logger=run)

  clf = train_classifier(ds)
  run.summary['FID'] = evaluate(learner, clf, ds)


if __name__ == "__main__":
  main()
