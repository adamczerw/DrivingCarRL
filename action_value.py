from abc import ABC, abstractmethod
import numpy as np
from enviroment import *

class ActionValue(ABC):
    def __init__(self, track):
        self.track = track
    
    @abstractmethod
    def get_value(self, state, action):
        pass

class TensorActionValue(ActionValue):
    """Action-value function estimation stored in a tensor. 
        Indices represent state and actions. 
        Zero values for not allowed state-action pairs.
        Works in this case beacuse actual action-values are non-zero (negative)."""
    def __init__(self, track):
        super().__init__(track)
        width, height = self.track.grid.shape 
        Q = np.zeros((width, height, 6, 6 , 3, 3)) # more elemens then there is (s,a) pairs, but we will just ignore them
        proper_states_height, proper_states_weight = np.where((self.track.grid == 0) | (self.track.grid==1))  # 0 - start, 1 - track
        proper_states = list(zip(proper_states_height, proper_states_weight))
        
        for i,j in proper_states:
            Q[i,j] = 1
            
        Q[:,:,0,:,0,:] = 0 # those elements represent exceeding speed limits
        Q[:,:,5,:,2,:] = 0
        Q[:,:,:,0,:,0] = 0
        Q[:,:,:,5,:,2] = 0

        Q[:,:,0,0,1,1] = 0 # those elements represent getting speed to zero for both directions
        Q[:,:,1,0,0,1] = 0
        Q[:,:,0,1,1,0] = 0
        Q[:,:,1,1,0,0] = 0
        
        Q = Q * (-width*height) # arbitrarily choosen starting values

        self.Q = Q

    def get_value(self, state, action):
        return self.Q[tuple(state+action)]
