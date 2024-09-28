import numpy as np
import math
import copy
import sys
from time import time 

class Track:
    def __init__(self, i, j):
        self.grid = np.ones((i, j))
    
    def __str__(self):
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
            self.final_position = [check[1], check[2]]


        
        
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
                return "finish", vertical_position, horizontal_position
        while vertical_speed:
            vertical_position -= 1
            vertical_speed -= 1
            if self._is_off_track(vertical_position, horizontal_position):
                return "boundry"
            if self._is_on_finish(vertical_position, horizontal_position):
                return "finish", vertical_position, horizontal_position
        while horizontal_speed:
            horizontal_position += 1
            horizontal_speed -= 1
            if self._is_off_track(vertical_position, horizontal_position):
                return "boundry"
            if self._is_on_finish(vertical_position, horizontal_position):
                return "finish", vertical_position, horizontal_position
        return "track"
    
    def _is_off_track(self, vertical_position = None, horizontal_position = None):
        if vertical_position is None:
            vertical_position = self.car.vertical_position
            horizontal_position = self.car.horizontal_position
        rows, cols = self.track.grid.shape
        return not (0 <= vertical_position < rows and 0 <= horizontal_position < cols) or (self.track.grid[vertical_position, horizontal_position] == -1)
        
    def _is_on_finish(self, vertical_position = None, horizontal_position = None):
        if vertical_position is None:
            vertical_position = self.car.vertical_position
            horizontal_position = self.car.horizontal_position
        return self.track.grid[vertical_position, horizontal_position] == 2

