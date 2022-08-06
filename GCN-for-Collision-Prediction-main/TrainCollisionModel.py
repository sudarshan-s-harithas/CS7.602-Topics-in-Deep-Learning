import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import LoadData as LD
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



TrainData = LD.GetData()

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(20, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN().to(device)
# data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(20):

	for i in range( len( TrainData ) - 5):
		data= TrainData[i]
		optimizer.zero_grad()
		out = model(data)
		loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
		loss.backward()
		optimizer.step()



model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
print(correct)

acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
torch.save(model, 'collision_detection_Batch_GNN.pt')

