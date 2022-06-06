

The repository consists of code to implement three solutions to Question 3 of the assignment. 

Q.3.1) The class that can support various forms of GNN and that implements the process of initilization, aggregation, combination, output can be found at the Models_Q3.py code. They are defined by the class GNN and GraphConvolution. 

Q.3.2) To run the gcn code please execute the command 

python3 Question3.py gcn

THe code implements a two layer GCN
GCN has been trained on citeseer and a accuracy of 71% is obtained. 

Q.3.3) As a variant of GNN we implement GraphSage to execute GraphSage please run the command

python3 Question3.py GraphSage

On citeseer dataset GraphSage works with an accuracy of 67%

Q.3.4)  To implement the vanilla RNN run the command 

python3 Question3.py rnn

The IMDB sentiment classification dataset is used, it performs with an accuracy of 58.5%

