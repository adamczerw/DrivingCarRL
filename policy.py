from abc import ABC, abstractmethod
import numpy as np
from enviroment import *
from action_value import *

class Policy(ABC):
    def __init__(self, track):
        self.track = track
    
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
    def __init__(self, track, Q):
        super().__init__(track)
        

        height, width = track.grid.shape
        self.policy = np.zeros((height, width, 6, 6, 3, 3))
        proper_states_height, proper_states_width = np.where((self.track.grid == 0) | (self.track.grid==1))  # 0 - start, 1 - track
        proper_states = list(zip(proper_states_height, proper_states_width))
        
        for i,j in proper_states:
            for k in range(6):
                for l in range(6):
                    #self.update_policy([i,j,k,l])
                    masked_Q = np.ma.masked_equal(Q[i,j,k,l], 0)
                    greedy_action_index = np.unravel_index(np.argmax(masked_Q), (3,3)) 
                    #self._policy[i,j,k,l] = 0
                    self.policy[i,j,k,l][greedy_action_index] = 1

    # def update_policy(self, state):
    #     masked_Q = np.ma.masked_equal(self.Q[tuple(state)], 0)
    #     greedy_action_index = np.unravel_index(np.argmax(masked_Q), self.Q.shape[-2:]) 
    #     self._policy[tuple(state)] = 0
    #     self._policy[tuple(state)][greedy_action_index] = 1
    def get_action(self, state):
        return list(np.argwhere(self.policy[tuple(state)]==1)[0]-1)

class EpsilonSoftPolicy(Policy):
    def __init__(self, track, epsilon, Q):
        super().__init__(track)
        self.epsilon = epsilon
        height, width = track.grid.shape
        proper_states_height, proper_states_width = np.where((self.track.grid == 0) | (self.track.grid==1))  # 0 - start, 1 - track
        proper_states = list(zip(proper_states_height, proper_states_width))
        self.policy = (Q[:] != 0).astype(int) * self.epsilon
        for i,j in proper_states:
            for k in range(6):
                for l in range(6):
                    As = np.count_nonzero(self.policy[i,j,k,l]) # number of possible actions
                    self.policy[i,j,k,l] = self.policy[i,j,k,l] / As
        greedy_policy = GreedyPolicy(track, Q)
        self.policy = self.policy + greedy_policy.policy - greedy_policy.policy * self.epsilon
        
    def get_action(self, state):
        action_distribution = self.policy[tuple(state)].flatten()
        selected_action = np.random.choice(len(action_distribution), p=action_distribution)
        action_index = np.unravel_index(selected_action, (3,3))
        return [i-1 for i in action_index]
    
    def get_propability(self, state, action):
        return self.policy[tuple(state+[i+1 for i in action])]

