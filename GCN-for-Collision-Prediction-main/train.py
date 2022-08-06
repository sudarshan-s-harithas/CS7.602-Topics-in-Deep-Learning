import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

mu1 = 0 
sigma1 = 2 




dist1 = np.random.normal(0, sigma1, 20)
dist2 = np.random.normal(0.6, sigma1, 20)
dist3 = np.random.normal(0.8, sigma1, 20)
dist4 = np.random.normal(1.2, sigma1, 20)

dist5 = np.random.normal(6.4, sigma1, 20)
dist6 = np.random.normal(2.5, sigma1, 20)
dist7 = np.random.normal(5, sigma1, 20)
dist8 = np.random.normal(6, sigma1, 20)
dist9 = np.random.normal(5.5, sigma1, 20)
dist10 = np.random.normal(8.2, sigma1, 20)

dist11 = dist1
dist12 = dist3
dist13 = dist6
dist14 = dist8
dist15 = dist9 
dist16 = dist1 
dist17 = dist8



Data1 = np.array( [ dist1 , dist2 , dist3 , dist4 , dist5 ,
dist6, dist7 , dist8 , dist9 , dist10 , dist11 , dist12 , dist13 , dist14 , dist15 ,dist16 , dist17 ])



edge_index = torch.tensor([[0, 10],
                           [10, 0],

                           [1, 10],
                           [10, 1] ,

                           [2 , 11] , 
                           [11, 2] , 

                           [3 , 11] , 
                           [11 , 3] , 

                           [4 , 12] , 
                           [12 , 4] ,

                           [5,12],
                           [12,5],

                           [6,13],
                           [13,6],

                           [7 , 13],
                           [13 , 7],

                           [8,14],
                           [14,8],

                           [9 ,14],
                           [14,8],

                           [10,15],
                           [15,10],

                           [15,11],
                           [11,15],

                           [12,16],
                           [16,12],

                           [13,16],
                           [16,13],

                           [14,16],
                           [16,14]

                            ] , dtype=torch.long)

edge_index = torch.transpose( edge_index ,  0 , 1)

y = torch.tensor( [ 1,0,0,0,0,1,1,1,0,0,1,0,1,1,0,1,1] )


data = Data(x=Data1, edge_index=edge_index , y=y)


print(data)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(  20 , 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
# data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out , y )
    loss.backward()
    optimizer.step()


# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='/tmp/Cora', name='Cora')

# print( dataset.num_node_features)
# data = dataset[0]

# print(data)