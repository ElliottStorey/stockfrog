import torch
from torch import nn

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"using {device} device")

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_leakyrelu_stack = nn.Sequential(
      nn.Linear(5, 25),
      nn.LeakyReLU(),
      nn.Linear(25, 512),
      nn.LeakyReLU(),
      nn.Linear(512, 25),
      nn.LeakyReLU(),
      nn.Linear(25, 5),
      nn.LeakyReLU(),
      nn.Linear(5, 1)
    )

  def forward(self, x):
    return self.linear_leakyrelu_stack(x)

model = Model().to(device)