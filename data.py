import torch
import yfinance as yf

class Dataset(torch.utils.data.Dataset):
  def __init__(self, ticker):
    ticker = yf.Ticker(ticker)
    self.history = ticker.history(interval='1m', period='max', prepost=True)
    print(self.history)

  def __len__(self):
    return len(self.history)

  def __getitem__(self, index):
    history = self.history.iloc[index]
    return torch.tensor((history.name.timestamp(), history.Open, history.High, history.Low, history.Volume), dtype=torch.float32), torch.tensor(history.Close, dtype=torch.float32)

train_data = Dataset('MSFT')
test_data = Dataset('MSFT')