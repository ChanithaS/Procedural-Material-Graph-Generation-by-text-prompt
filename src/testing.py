import torch
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

device = torch.device("mps")

### Import data
dataset = Planetoid(root="tmp/Cora", name="Cora")
data = dataset[0]
print("Cora dataset:", data)

n2v_model = Node2Vec(edge_index=data.edge_index, embedding_dim=128, walk_length=10, 
                    context_size=10, walks_per_node=10, p=1, q=1, sparse=True).to(device)

def create_loader(model):
    return model.loader(batch_size=64, shuffle=True, num_workers=4)

optimizer = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

if __name__ == '__main__':
    loader = create_loader(n2v_model)
    loss=[]
    n2v_model.train()
    for sample in loader:
        print(sample)
        optimizer.zero_grad()
        break