import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from fid import compute_fid


class ImageDS(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.image_files = [str(f.relative_to(root_dir)) for f in Path(root_dir).rglob('*') if f.suffix.lower() in ('.png', '.jpg')]

    self.transform = transforms.Compose([
        transforms.Resize(342, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_path = Path(self.root_dir) / self.image_files[idx]
    image = read_image(str(img_path))
    if image.shape[0] == 1:
      return self.__getitem__(idx+1)
    image = self.transform(image)
    return image


if __name__ == "__main__":
  ds = ImageDS('imagenet_samples')
  # test_ds = ImageDS('generated_images/runwayml--stable-diffusion-v1-5')
  test_ds = ImageDS('generated_images/black-forest-labs--FLUX.1-dev')

  dl = DataLoader(ds, batch_size=32, shuffle=True)
  test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

  model = inception_v3(weights=Inception_V3_Weights)
  model.fc = torch.nn.Identity()
  model.eval()
  model.to(device)

  all_real = []
  all_gen = []

  with torch.no_grad():
    for batch in tqdm(dl):
      all_real.append(model(batch.to(device)).cpu())

    for batch in tqdm(test_dl):
      all_gen.append(model(batch.to(device)).cpu())

  out_real = torch.cat(all_real)
  out_gen = torch.cat(all_gen)

  print(compute_fid(out_gen, out_real))
  # 74 for sd 1.5
  # 77 for flux
