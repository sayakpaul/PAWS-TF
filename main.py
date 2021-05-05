from utils import resnet20

print(resnet20.get_network(n=2, hidden_dim=128).summary())