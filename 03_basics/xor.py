import torch
from torch import nn
from tqdm import trange


class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_layer = nn.Linear(2, 64)
    self.hidden_layer = nn.Linear(64, 64)
    self.output_layer = nn.Linear(64, 1)

  def forward(self, x):
    for _, layer in enumerate([self.input_layer, self.hidden_layer, self.output_layer]):
      x = layer(x)
      x = torch.nn.functional.relu(x)
    return x

nn = NeuralNet()
dummy_inputs = torch.randn(2).unsqueeze(0)
nn(dummy_inputs)

optim = torch.optim.Adam(nn.parameters(), lr=1e-3) # params -= gradients * lr

def loss_fn(y, y_hat):
  return torch.square(y-y_hat).mean()

X = torch.Tensor(
  [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
  ]
)

Y = torch.Tensor(
  [
    [0],
    [1],
    [0],
    [1],
  ]
)

for _ in trange(1000):
  for x, y in zip(X, Y):
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    y_hat = nn(x)
    loss = loss_fn(y, y_hat)
    loss.backward()
    optim.step()
    optim.zero_grad()

for x in X:
  print(x, nn(x.unsqueeze(0)))
