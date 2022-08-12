import torch
import numpy as np
from traitlets import Float

class Test(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(1, 1000)
        self.linear_2 = torch.nn.Linear(1000, 1000)
        self.linear_3 = torch.nn.Linear(1000, 1)
        
    def forward(self, x):
        y = self.linear_1(x)
        y = torch.tanh(y)
        y = self.linear_2(y)
        y = torch.tanh(y)
        y = self.linear_3(y)
        y = torch.sigmoid(y)

        return y

x = np.arange(-np.pi, np.pi, 0.001, np.float32)
y = np.sin(x)
x = torch.from_numpy(x).view(-1, 1)
y = torch.from_numpy(y).view(-1, 1)

test = Test()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(test.parameters())
for epoch in range(1000):
    y_hat = test(x)
    loss = loss_function(y, y_hat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("epoch", epoch, loss.item())

print(test(torch.tensor(0.0, dtype=torch.float32).view(-1,1)))