import torch 
import torch.nn as nn

class C3(nn.Module):
    def __init__(self, connection_table, in_maps =6, out_maps = 16, kernel_size = 5):
        super(C3, self).__init__()
        self.connection_table = torch.tensor(connection_table, dtype=torch.uint8)
        assert self.connection_table.shape == (in_maps, out_maps)

        self.convs = nn.ModuleList()
        for j in range(out_maps):
            in_ch = int(self.connection_table[:, j].sum().item())
            self.convs.append(
                nn.Conv2d(in_ch, 1, kernel_size=kernel_size, stride=1, padding=0)
            )

    def forward(self, x):
        outputs = []
        for j, conv in enumerate(self.convs):
            connected = self.connection_table[:, j].bool()
            subset = x[:, connected, :, :]
            y = conv(subset)
            outputs.append(y)
        return torch.cat(outputs, dim=1)   # → (batch, 16, 10, 10)