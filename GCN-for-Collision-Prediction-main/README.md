# GNN-for-Collision-Prediction

This work explors the use of a GNN to perform collision prediction of trajectories. 


### Simulation Results 
Create a folder name **Plots** before the start of the code to save results

The simulation can be run using the following command 

```
python3 run_Linearized_VO.py
```

The results of the prediction would be stored in the folder named *Plots*  the resulting images can be converted to a video for ease of visulization using the below command 


```
python3 Image2Video.py
```

The resulting simulation would appear as shown below 

![](https://github.com/sudarshan-s-harithas/GNN-for-Collision-Prediction/blob/main/Images/TDL_project2.gif)



### Train model from scratch 

The data that is required for trainning is provided as a part of this code repository, please run the command below to train the model from scratch. 

```
python3 TrainCollisionModel.py
```
