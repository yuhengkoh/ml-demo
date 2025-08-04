import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 1000)).to(device)
data = torch.randn(1024, 1000).to(device)
target = torch.randn(1024, 1000).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

for _ in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    print(loss.item())
