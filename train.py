import torch
import torch.nn as nn
from module import BatchNormRelu 


class KernelFusedModel(nn.Module):
    def __init__(self):
        super(KernelFusedModel, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn_relu = BatchNormRelu(channels=64, resolution=128)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_relu2 = BatchNormRelu(channels=128, resolution=64)

        self.linear = nn.Linear(in_features=128*64*64, out_features=10)

    def forward(self, x):  
        x = self.conv1(x)
        x = self.bn_relu(x)
        x = self.conv2(x)
        x = self.bn_relu2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear(x)
        return x
    

#Example training run
if __name__ == "__main__":
    device = "cuda"

    training_data = torch.rand((32, 3, 256, 256)).to(device)
    ground_truths = torch.rand((32, 10)).to(device)

    model = KernelFusedModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        res = model(training_data)
        loss = nn.MSELoss()(res, ground_truths)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

