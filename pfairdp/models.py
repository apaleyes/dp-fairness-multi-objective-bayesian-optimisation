import torch

# Model used for evaluations on ADULT
class SNNSmall(torch.nn.Module):
    def __init__(self, num_features):
        super(SNNSmall, self).__init__()
        
        self.hid1 = torch.nn.Linear(num_features, 6)
        self.hid2 = torch.nn.Linear(6, 6)
        self.out = torch.nn.Linear(6, 1)
        
    def forward(self, x):
        z = torch.nn.functional.relu(self.hid1(x)) 
        z = torch.nn.functional.relu(self.hid2(z))
        z = torch.sigmoid(self.out(z)) 
        return z

# Model used for evaluations on MEPS
class SNNMedium(torch.nn.Module):
    def __init__(self, num_features):
        super(SNNMedium, self).__init__()
        
        self.hid1 = torch.nn.Linear(num_features, 30)
        self.hid2 = torch.nn.Linear(30, 30)
        self.out = torch.nn.Linear(30, 1)
        
    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.sigmoid(self.out(z)) 
        return z

# Model used for comparison with Xu et al.
class EquivalentLogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(EquivalentLogisticRegression, self).__init__()

        self.fc1 = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))

        return out