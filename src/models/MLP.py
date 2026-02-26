import torch



class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(torch.nn.ReLU())
            in_size = h
        layers.append(torch.nn.Linear(in_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
