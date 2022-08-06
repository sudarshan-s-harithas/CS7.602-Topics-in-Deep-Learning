
import numpy as np
import math
import draw
import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from scipy.optimize import minimize


import frenet_optimal_trajectory
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

wx = [0.0, 120.0]
wy = [0.0,0.0]
tx, ty, tyaw, tc, csp = frenet_optimal_trajectory.generate_target_course(wx, wy)

print(csp)

wx=np.array(wx)
wy=np.array(wy)  
# initial state
c_speed = 1.5*6  # current speed [m/s]
c_d = 0.0  # current lateral position [m]
c_d_d = 0.0  # current lateral speed [m/s]
c_d_dd = 0.0  # current latral acceleration [m/s]
s0 = 0.0  # current course position
#%========================================%
#%Simulation details
#%========================================%
agent_r = 0.5*4
obs_r = 0.5*4
goal_r = 0.5*4
vel_cap_max = 1.5
vel_cap_min = 0.3
w_cap = 0.75
dt = 0.1
sensor_range = 7.
lb =[-0.2, -0.2]
ub = [0.2, 0.2]
ub_w = ub[1]
lb_w = lb[1]
ub_v = ub[0]
lb_v = lb[0]
u0 = [0,0]
#%Collision cone without linearization
#%% Simulation parameters for the robot, obstacle and goal
Case = 0.
agent_p = np.array([0,0])
#obs_p = np.array([[5,-0.1],[11,1.5],[13,1.33],[13.5,3.7],[15,3.5],[ 13,0],[7,-1.8],[10,-1.7]])
#obs_v = np.array([[0.3,0],[-0.43,0],[-0.23,0],[-0.313,0],[-0.253,0],[0.3,0],[0.27,0],[0.42,0]])
obs_p = np.array([[7.5*2,0],[9*2,-2],[8*2,-2],[7*2,1.5],[9*2,2], [25*2,0],[ 12*2 ,-1.5], [14*2, 2.4]])*3
obs_v = np.array([[-0.5,0],[0.4,0],[0.3,0],[-0.25,0],[0.4,0],[-0.6,0],[0.5,0],[0.5,0]])*6
goal_p = np.array([120,0])
#%%
#%========================================%
#%Simulation details continued
#%========================================%
CollidingTraj =0 
TotalTraj =0 

x_min = np.min(np.hstack((obs_p[:,0],agent_p[0],goal_p[0])))
x_max = np.max(np.hstack((obs_p[:,0],agent_p[0],goal_p[0])))
y_min = np.min(np.hstack((obs_p[:,1],agent_p[1],goal_p[1])))
y_max = np.max(np.hstack((obs_p[:,1],agent_p[1],goal_p[1])))
range = np.max(np.array([x_max-x_min, y_max-y_min]))
x_axis_min = (x_min+x_max)/2.-range/2.-2.
x_axis_max = (x_min+x_max)/2.+range/2.+2.
y_axis_min = (y_min+y_max)/2.-range/2.-2.
y_axis_max = (y_min+y_max)/2.+range/2.+2.
current_head = math.atan2((goal_p[1]-agent_p[1]), (goal_p[0]-agent_p[0]))
#%variable to store current velocity direction
prev_rel_p = []
current_rel_p = []
agent_v = np.array([1.5,0])
num_of_obs = len(obs_p)
bot_path = np.array([agent_p])
#%Varable to store robot's traces to plot it's path
obs_path = []
#%Varable to store obstacle's traces to plot it's path

#%Varable to store obstacle samples' traces to plot it's path
prev_agent_p = agent_p
prev_obs_p = obs_p
prev_rel_p = obs_p-agent_p
counter = 1.
save=0
#%% Start of simulation


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

model = torch.load('collision_detection_Batch_GNN_max.pt')
model.eval()  


global_trajectory_cntr = 0 


while np.linalg.norm((agent_p-goal_p)) > 0.2:
    current_head = current_head+np.dot(agent_v[1], dt)
    #%updating our current heading
    v = np.array([np.dot(agent_v[0], np.cos(current_head)), np.dot(agent_v[0], np.sin(current_head))])
    #%velocity value in x and y direction from v and w controls
    prev_vec=agent_p;
#    agent_p = agent_p+np.dot(v, dt)
    
    #%updating robots position
    bot_path=np.append(bot_path,[agent_p],axis=0)
    #%storing robots path   
    obs_p = obs_p+np.dot(obs_v, dt)
    #%updating obstacles position
    #obs_path[:,:,int(counter)-1] = obs_p
    #%storing obstacles path
    current_rel_p = obs_p-agent_p
    #%relative position
    current_rel_v = (current_rel_p-prev_rel_p)/dt
    #%relative velocity
    lb = np.array([lb_v,lb_w])
    ub = np.array([ub_v,ub_w])
    if agent_v[0] > vel_cap_max-ub[0]:
        ub[0] = vel_cap_max-agent_v[0]
    elif agent_v[0]<vel_cap_min-lb[0]:
        lb[0] = -(agent_v[0]-vel_cap_min)
        
    
    if agent_v[1] > w_cap-ub[1]:
        ub[1] = w_cap-agent_v[1]
    elif agent_v[1]<-w_cap-lb[1]:
        lb[1] = -(agent_v[1]+w_cap)

    
    vr = agent_v[0]
    wr = agent_v[1]
    xr = agent_p[0]
    yr = agent_p[1]
    xob = obs_p[:,0].conj().T
    yob = obs_p[:,1].conj().T
    xobdot = obs_v[:,0].conj().T
    yobdot = obs_v[:,1].conj().T
    R = agent_r+obs_r
    detected = np.empty(0)
    infront_condition = 0.
    for l in np.arange(num_of_obs):
        infront_condition = np.dot(current_rel_p[l,:], current_rel_v[l,:])<0.
        distance_condition = np.linalg.norm((agent_p-obs_p[l,:]))<sensor_range
        if distance_condition and infront_condition:
            detected=np.append(detected,int(l))        
        

    v_desired = np.dot(vel_cap_max, goal_p-agent_p)/np.linalg.norm((goal_p-agent_p))
    objective = lambda u: np.linalg.norm(v_desired-np.dot(agent_v[0]+u[0], np.array([np.cos(current_head+np.dot(dt, agent_v[1]+u[1])), np.sin(current_head+np.dot(dt, agent_v[1]+u[1]))])))
    detected=detected.astype(int)
    xob = xob[detected]
    yob = yob[detected]
    xobdot = xobdot[detected]
    yobdot = yobdot[detected]
    ob=np.hstack((xob,yob))
    ob_v=np.hstack((xobdot,yobdot))

    path , global_trajectory_cntr = frenet_optimal_trajectory.frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, obs_p,obs_v,prev_vec , model , global_trajectory_cntr )

    s0 = path.s[1]
    c_d = path.d[1]
    c_d_d = path.d_d[1]
    c_d_dd = path.d_dd[1]
    c_speed = path.s_d[1]
    
    '''
    if detected.size:
        xob = xob[detected]
        yob = yob[detected]
        xobdot = xobdot[detected]
        yobdot = yobdot[detected]
        print(agent_v)        
        print(ub,lb)
        opt = optimize.optimize(current_head, vr, wr, xr, yr, xob, yob, xobdot, yobdot, R, dt, lb, ub, objective)
    else:
        bnds = ((lb[0], ub[0]), (lb[1], ub[1]))
        opt = minimize(objective,(0,0), method='SLSQP', bounds=bnds)
        opt=opt.x
    '''    
    #print( obs_p)    
    agent_v[0] = path.v[0]
    agent_v[1] = path.w[0]
    agent_p=[path.x[0],path.y[0]]
    print(agent_v, s0, c_speed)
    #%updating the controls
    prev_rel_p = current_rel_p
    draw.draw(obs_p, agent_p,v, goal_p, obs_r, agent_r, goal_r, bot_path,x_axis_min, x_axis_max, y_axis_min, y_axis_max,counter,dt,save,path)
    counter = counter+1.
    
