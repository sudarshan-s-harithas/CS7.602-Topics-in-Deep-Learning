import numpy as np 
import torch
from torch_geometric.data import Data

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

train_mask = torch.tensor( [ True, True , True, False,True,True,True,True,True,False,False,False,False,False,False,False,False])

edge_index = torch.transpose( edge_index ,  0 , 1)

def GetBatches(  BatchNum, DataArray , Labels , BatchSize ):

	BatchData  = np.zeros( ( BatchSize  , 20 ) )
	BatchLabel = np.zeros((BatchSize , 1 ) )

	for i in range(10):

		BatchData[i] = DataArray[ 10*BatchNum + i]
		BatchLabel[i] = Labels[ 10*BatchNum + i ]

	cntr = 10

	for i in range( 0, 10, 2):

		if(BatchLabel[i] == 1 ):
			BatchData[cntr] =  BatchData[i]
			BatchLabel[cntr] = 1 

		if( BatchLabel[i+1] == 1 ):
			BatchData[cntr] =  BatchData[i+1]
			BatchLabel[cntr] = 1 

		if( BatchLabel[i+1] == 0 and BatchLabel[i] == 0  ):

			BatchData[cntr] =  BatchData[i+1]
			BatchLabel[cntr] = 0 

		cntr +=1 


	if( BatchLabel[10] == 1 or BatchLabel[11] == 1):
		BatchLabel[15] =1 
		BatchData[15] = BatchData[10]

	if( BatchLabel[10] == 0 and BatchLabel[11] == 0 ):
		BatchLabel[15] =0 
		BatchData[15] = BatchData[10]


	if( BatchLabel[12] == 1 or BatchLabel[13] == 1 or BatchLabel[14] == 1 ):
		BatchLabel[16] =1
		BatchData[16] = BatchData[13]



	y = torch.tensor( np.reshape(BatchLabel,  (17,)) ,   device='cpu' , dtype=torch.long )
	Data1 = torch.tensor(BatchData , device='cpu' , dtype=torch.float32)
	data = Data(x=Data1, edge_index=edge_index , y= y , train_mask=train_mask , val_mask= train_mask, test_mask=train_mask)

	return data



def GetData():

	DataArray = np.loadtxt("CollisionData_Max.txt").reshape(200, 20)  #np.loadtxt("CollisionData.txt").reshape(200, 20)
	Labels = np.loadtxt("Labels_Max.txt").reshape(200, 1)#np.loadtxt("Labels.txt").reshape(200, 1)

	DataIn = []

	numBatches = 20 
	BatchSize = 17 

	for i in range( numBatches ):

		Batchdata = GetBatches( i , DataArray , Labels , BatchSize )
		DataIn.append( Batchdata )

	return DataIn





def CreateFFEdgeList( numObs, numPtsPerTraj):

	edge_list = []

	total_nodes = numObs+ numPtsPerTraj

	for i in range(numObs):

		for j in range(numObs , numObs+numPtsPerTraj, ):

			edge = [ i,j ]
			edge_list.append(edge)


	for i in range(numPtsPerTraj):

		edge = [ i, total_nodes]
		edge_list.append(edge)


	edge_index = torch.tensor( edge_list ,  dtype=torch.long)


	return edge_index




# def GetBatchDataArchB( BatchNum , DataArray , Labels , BatchSize  ):

# 	BatchData = np.zeros( (BatchSize , 20 ) )
# 	BatchLabel = np.zeros((BatchSize , 1 ) )


# def GetDataArchB( numObs, numPtsPerTraj ):

# 	edge_index = CreateFFEdgeList( numObs, numPtsPerTraj)

# 	DataIn = []

# 	DataArray = np.loadtxt("CollisionData_archB.txt").reshape(2000, 20)
# 	Labels = np.loadtxt("Labels_archB.txt").reshape(2000, 1)

# 	numBatches = 50
# 	BatchSize = numObs + numPtsPerTraj +1 

# 	for i in range(numBatches):

# 		BatchData = GetBatchDataArchB( i , DataArray , Labels , BatchSize )






