import torch
from torch import nn
from model import model
from data import train_data, test_data

learning_rate = 1e-3
batch_size = 64
epochs = 50

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  # Set the model to training mode - important for batch normalization and dropout layers
  # Unnecessary in this situation but added for best practices
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    y = y.unsqueeze(1)
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 20 == 0:
      loss, current = loss.item(), batch * batch_size + len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

test_losses = []

def test_loop(dataloader, model, loss_fn):
  # Set the model to evaluation mode - important for batch normalization and dropout layers
  # Unnecessary in this situation but added for best practices
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
  # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      y = y.unsqueeze(1)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()

  test_loss /= num_batches
  correct /= size
  test_losses.append(test_loss)
  print(f"avg test loss: {test_loss:>8f}\n")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train_loop(train_dataloader, model, loss_fn, optimizer)
  test_loop(test_dataloader, model, loss_fn)
print("Done!")

print(test_losses)
from matplotlib import pyplot as plt
plt.plot(test_losses)
plt.show()