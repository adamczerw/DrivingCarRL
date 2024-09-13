from abc import ABC, abstractmethod
from ex_5_12 import *

class Policy(ABC):
    def __init__(self, track):
        self.track = track
    
    # @abstractmethod
    # def get_action_distribution(self, state):
    #     pass
    
    @abstractmethod
    def get_action(self, state):
        pass
    
    def generate_episode(self):
        race = Race(self.track)
        episode = []
        iter_counter = 0
        while not race.is_finish:
            vertical_position, horizontal_position = race.car.vertical_position, race.car.horizontal_position
            vertical_speed = race.car.vertical_speed
            horizontal_speed = race.car.horizontal_speed
            vertical_change, horizontal_change = self.get_action([vertical_position, horizontal_position, vertical_speed, horizontal_speed])
            episode.append([vertical_position, horizontal_position, vertical_speed, horizontal_speed, vertical_change, horizontal_change]) # reward is always -1
            race.take_action(vertical_change, horizontal_change)     
            #print(episode[iter_counter])
            iter_counter += 1
            if iter_counter > 1000000:
                sys.exit("Error: The episode did not converge.")
            return episode

# class TensorPolicy(Policy):
#     def __init__(self, track):
#         super().__init__(track)
#         self.policy = None

class GreedyPolicy(Policy):
    def __init__(self):

    

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


class Algorithm(ABC):
    def __init__(self, track):
        self.track = track
        self.Q = None
        self.t_policy = None
    @abstractmethod
    def train(self):
        pass

