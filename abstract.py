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

class ActionValue(ABC):
    @abstractmethod
    def get_value(self, state, action):
        pass

class Algorithm(ABC):
    def __init__(self, track):
        self.track = track
        self.Q = None
        self.t_policy = None
    @abstractmethod
    def train(self):
        pass

