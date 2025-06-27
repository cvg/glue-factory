import torch
import torch.nn as nn
class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))

    def forward(self, x):
        # import pdb;pdb.set_trace()
        return torch.cat([head(x) for head in self.heads], dim=1)


class AngleDistanceHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(AngleDistanceHead, self).__init__()

        m = int(input_channels/4)

        heads = []
        for output_channels in sum(head_size, []):
            if output_channels != 2:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
            else:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        CosineSineLayer(m)
                    )
                )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)