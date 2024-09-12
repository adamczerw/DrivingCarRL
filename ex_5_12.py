import numpy as np
import math
import copy
import sys
from time import time 

class Track:
    def __init__(self, i, j):
        self.grid = np.ones((i, j))
    
    def __str__(self):
        #return str(self.grid)
        view = self.grid.astype(str)
        view[self.grid==0] = "-"
        view[self.grid==2] = "|"
        view[self.grid==1] = " "
        view[self.grid==-1] = "X"
        return str(view)
    
    def set_start(self, i, j):
        for k in range(len(i)):
            self.grid[i[k], j[k]] = 0

    def set_finish(self, i, j):
        for k in range(len(i)):
            self.grid[i[k], j[k]] = 2
    
    def set_boundries_manual(self, i, j):
        for k in range(len(i)):
            self.grid[i[k], j[k]] = -1
    
    def set_boundries_rectangle(self, i, j):
        self.grid[i[0]:i[1], j[0]:j[1]] = -1
    

class Car:
    def __init__(self, vertical_position = 0, horizontal_position = 0, vertical_speed = 0, horizontal_speed = 0):
        self.vertical_position = vertical_position
        self.horizontal_position = horizontal_position
        self.horizontal_speed = horizontal_speed
        self.vertical_speed = vertical_speed
    
    def change_speed(self, vertical, horizontal):
        self.vertical_speed += vertical
        self.horizontal_speed += horizontal
    
    # def set_position(self, position):
    #     self.position = position
    
    def drive(self):
        self.vertical_position -= self.vertical_speed
        self.horizontal_position += self.horizontal_speed
    
class Race:
    def __init__(self, track):
        self.track = track
        self.car = Car()
        self._put_car_on_start()
        self.score = 0
        self.is_finish = False
    
    def __str__(self):
        view = self.track.grid.astype(str)
        view[self.track.grid==0] = "-"
        view[self.track.grid==2] = "|"
        view[self.track.grid==1] = " "
        view[self.track.grid==-1] = "X"
        view[self.car.vertical_position, self.car.horizontal_position] = "C"
        return str(view)
    
    def _put_car_on_start(self):
        possible_starting_postions = np.argwhere(self.track.grid == 0)
        position = possible_starting_postions[np.random.randint(len(possible_starting_postions))]
        self.car.vertical_position, self.car.horizontal_position = position
        self.car.vertical_speed, self.car.horizontal_speed = 0, 0
    
    def take_action(self, vertical, horizontal):
        self.car.change_speed(vertical, horizontal)
        check = self._path_check()
        if check == "track":
            self.car.drive()
            self.score -= 1
        elif check == "boundry":
            self._put_car_on_start()
            self.score -= 1
        else:
            self.is_finish = True


        
        
    def _path_check(self):
        """
        Checks if the current path of the car goes outside of the track or crosses the finish line

        Returns:
            "boundry" if the car will cross the boundry
            "finish" if the car will cross the finish line
            "track" if it stays on the track
        """
        vertical_speed = int(self.car.vertical_speed)
        horizontal_speed = int(self.car.horizontal_speed)
        vertical_position = self.car.vertical_position
        horizontal_position = self.car.horizontal_position
        while vertical_speed and horizontal_speed:
            vertical_position -= 1
            horizontal_position += 1
            vertical_speed -= 1
            horizontal_speed -= 1
            if self._is_off_track(vertical_position, horizontal_position):
                return "boundry"
            if self._is_on_finish(vertical_position, horizontal_position):
                return "finish"
        while vertical_speed:
            vertical_position -= 1
            vertical_speed -= 1
            if self._is_off_track(vertical_position, horizontal_position):
                return "boundry"
            if self._is_on_finish(vertical_position, horizontal_position):
                return "finish"
        while horizontal_speed:
            horizontal_position += 1
            horizontal_speed -= 1
            if self._is_off_track(vertical_position, horizontal_position):
                return "boundry"
            if self._is_on_finish(vertical_position, horizontal_position):
                return "finish"
        return "track"
    
    def _is_off_track(self, vertical_position = None, horizontal_position = None):
        if vertical_position is None:
            vertical_position = self.car.vertical_position
            horizontal_position = self.car.horizontal_position
        rows, cols = self.track.grid.shape
        return (self.track.grid[vertical_position, horizontal_position] == -1) or not (0 <= vertical_position < rows and 0 <= horizontal_position < cols)
        
    def _is_on_finish(self, vertical_position = None, horizontal_position = None):
        if vertical_position is None:
            vertical_position = self.car.vertical_position
            horizontal_position = self.car.horizontal_position
        return self.track.grid[vertical_position, horizontal_position] == 2

def generate_episode(track, policy):
    race = Race(track)
    episode = []
    iter_counter = 0
    while not race.is_finish:
        vertical_position, horizontal_position = race.car.vertical_position, race.car.horizontal_position
        vertical_speed = race.car.vertical_speed
        horizontal_speed = race.car.horizontal_speed
        action_distribution = policy[vertical_position, horizontal_position, vertical_speed, horizontal_speed].flatten()
        #print(action_distribution, np.sum(action_distribution))
        selected_action = np.random.choice(len(action_distribution), p=action_distribution)
        vertical_change, horizontal_change = np.unravel_index(selected_action, (3,3))
        episode.append([vertical_position, horizontal_position, vertical_speed, horizontal_speed, vertical_change, horizontal_change]) # reward is always -1
        race.take_action(vertical_change - 1, horizontal_change - 1)     
        #print(episode[iter_counter])
        iter_counter += 1
        if iter_counter > 1000000:
             sys.exit("Error: The episode did not converge.")
    return episode

def off_policy_MC_control(track, n_iter = 100, behaviour_eps = 0.01):
    
    #--------------- Initializing Q (action value estimation) -----------------

    width, height = track.grid.shape 
    Q = np.zeros((width, height, 6, 6 , 3, 3)) # more elemens then there is (s,a) pairs, but we will just ignore them
    proper_states_height, proper_states_weight = np.where((track.grid == 0) | (track.grid==1))  # 0 - start, 1 - track
    proper_states = list(zip(proper_states_height, proper_states_weight))
    #proper_states2 = copy.deepcopy(proper_states) # a bit stupid I know, but I will be iterationg over proper_states twice
    
    for i,j in proper_states:
        Q[i,j] = 1
        Q[i,j,0,:,0,:] = 0 # those elements represent exceeding speed limits
        Q[i,j,5,:,2,:] = 0
        Q[i,j,:,0,:,0] = 0
        Q[i,j,:,5,:,2] = 0
    Q[:,:,0,0,1,1] = 0 # those elements represent getting speed to zero for both directions
    Q[:,:,1,0,0,1] = 0
    Q[:,:,0,1,1,0] = 0
    Q[:,:,1,1,0,0] = 0
    Q = Q * (-track.grid.shape[0]*track.grid.shape[1]) # arbitrarily choosen starting values

    #-------------- Template for the future behaviour policies ---------------------
    b_policy_template = Q / ((-track.grid.shape[0]*track.grid.shape[1])) * behaviour_eps 
    for i,j in proper_states:
        for k in range(6):
            for l in range(6):
                As = np.count_nonzero(b_policy_template[i,j,k,l]) # number of possible actions
                b_policy_template[i,j,k,l] = b_policy_template[i,j,k,l] / As
    #----------- Initializing cummulative sum of weights ----------------------------

    C  = np.zeros((width, height, 6, 6 , 3, 3)) 

    #--------------- Initializing target policy -------------------------------

    t_policy = np.zeros((width, height, 6, 6, 3, 3))
    for i,j in proper_states:
        for k in range(6):
            for l in range(6):
                non_zero_indices = np.argwhere(Q[i,j,k,l] != 0)
                random_index = non_zero_indices[np.random.choice(non_zero_indices.shape[0])]
                t_policy[i,j,k,l, random_index[0], random_index[1]] = 1
                
                # greedy_action_index = np.unravel_index(np.argmax(-Q[i,j,k,l]), Q.shape[-2:]) 
                # t_policy[i,j,k,l][greedy_action_index] = 1
    
    # t_policy[3,0,0,0,:,:] = 0
    # t_policy[3,0,0,0,2,1] = 1 # to speed up 
    # t_policy[3,1,0,0,:,:] = 0
    # t_policy[3,1,0,0,2,1] = 1

    #--------------- Main loop ------------------------------------------------
    episodes = []
    start = time()
    for i in range(n_iter):
        b_policy = b_policy_template.copy()
        b_policy = b_policy + t_policy - t_policy * behaviour_eps # adjusting probabilities for the greedy actions
        episode = generate_episode(track, b_policy)
        episodes.append(episode)
        G = 0
        W = 1
        iter_counter = 0
        for t in range(len(episode)-1, -1, -1):
            iter_counter += 1
            G += -1
            #print(tuple(episode[t]))
            C[tuple(episode[t])] += W
            Q[tuple(episode[t])] += W / C[tuple(episode[t])] * (G - Q[tuple(episode[t])]) 
            #print(Q[tuple(episode[t])])
            
            
            masked_Q = np.ma.masked_equal(Q[tuple(episode[t][:-2])], 0)
            greedy_action_index = np.unravel_index(np.argmax(masked_Q), Q.shape[-2:]) 
            t_policy[tuple(episode[t][:-2])] = 0
            t_policy[tuple(episode[t][:-2])][greedy_action_index] = 1
            
            # if t == 0:
            #     print(t)
            #print(episode[t][-2:], list(greedy_action_index))
            if episode[t][-2:] != list(greedy_action_index):
                # if iter_counter == 1:
                #     print("break po ostatnim zdarzeniu z epizodu")
                # else:
                #     print("break NIE po ostatnim zdarzeniu z epizodu")
                break
            W = W / b_policy[tuple(episode[t])]
        end = time() 
        print(f"Iteration: {i}, seconds elapsed: {round(end-start, 2)}")
        start = time()
    #--------------------------------------------------------------------------
        
    return Q, t_policy, episodes

def main():
    droga = Track(4, 4)
    droga.set_start((np.ones(2) * 3).astype(int), np.arange(2))
    droga.set_finish(np.arange(2), (np.ones(2) * 3).astype(int))
    droga.set_boundries_rectangle([2,4], [2,4])
    print(droga)
    Q, t_policy = off_policy_MC_control(droga, n_iter = 1)
    

if __name__ == "__main__":
    main()