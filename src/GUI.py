import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
from .algorithms import OffPolicyMCControl

class Track:
    def __init__(self, height, width):
        self.grid = np.ones((height, width))  # Initialize grid with 1
    
    def set_start(self, x, y):
        if self.grid[x, y] == 0:
            self.grid[x, y] = 1  # Reset to default state if already start
        else:
            self.grid[x, y] = 0  # Set start position to 0
    
    def set_finish(self, x, y):
        if self.grid[x, y] == 2:
            self.grid[x, y] = 1  # Reset to default state if already finish
        else:
            self.grid[x, y] = 2  # Set finish position to 2
    
    def set_boundary(self, x, y):
        if self.grid[x, y] == -1:
            self.grid[x, y] = 1  # Reset to default state if already boundary
        else:
            self.grid[x, y] = -1  # Set boundary position to -1
    
    def reset_cell(self, x, y):
        self.grid[x, y] = 1  # Reset cell to default state
    
    def set_boundaries_rectangle(self, x_range, y_range):
        self.grid[x_range[0]:x_range[1], y_range[0]:y_range[1]] = -1
    
    def clear(self):
        self.grid.fill(1)  # Reset grid to 1
    
    def __repr__(self):
        return str(self.grid)

class TrackDesigner:
    def __init__(self, root):
        self.root = root
        self.track = None
        self.canvas = None
        self.rect_start = None
        self.rect_id = None
        
        self.create_layout()
        self.create_initial_controls()
    
    def create_layout(self):
        self.left_frame = tk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="n")
        
        self.middle_frame = tk.Frame(self.root)
        self.middle_frame.grid(row=0, column=1)
        
        self.right_frame = tk.Frame(self.root)
        self.right_frame.grid(row=0, column=2, sticky="n")
    
    def create_initial_controls(self):
        self.height_label = tk.Label(self.left_frame, text="Height:")
        self.height_label.grid(row=0, column=0, padx=5, pady=5)
        self.height_entry = tk.Entry(self.left_frame)
        self.height_entry.insert(0, "15")
        self.height_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.width_label = tk.Label(self.left_frame, text="Width:")
        self.width_label.grid(row=1, column=0, padx=5, pady=5)
        self.width_entry = tk.Entry(self.left_frame)
        self.width_entry.insert(0, "10")
        self.width_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.create_button = tk.Button(self.left_frame, text="Create Grid", command=self.create_grid)
        self.create_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    def create_grid(self):
        height = int(self.height_entry.get())
        width = int(self.width_entry.get())
        self.track = Track(height, width)
        
        if self.canvas:
            self.canvas.destroy()
        
        self.canvas = tk.Canvas(self.middle_frame, width=width*20, height=height*20)
        self.canvas.pack()
        self.draw_grid()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        
        self.mode = tk.StringVar(value="start")
        self.create_controls()
        
        # Hide initial controls
        self.height_label.grid_forget()
        self.height_entry.grid_forget()
        self.width_label.grid_forget()
        self.width_entry.grid_forget()

        # Remove the "Create Grid" button
        self.create_button.destroy()
    
    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.track.grid.shape[0]):
            for j in range(self.track.grid.shape[1]):
                color = "white"
                if self.track.grid[i, j] == 0:
                    color = "green"  # Start position
                elif self.track.grid[i, j] == 2:
                    color = "red"  # Finish position
                elif self.track.grid[i, j] == -1:
                    color = "black"  # Boundary
                elif self.track.grid[i, j] == 3:
                    color = "blue"  # Car
                elif self.track.grid[i, j] == 1:
                    color = "white"  # Normal grid
                self.canvas.create_rectangle(j*20, i*20, (j+1)*20, (i+1)*20, fill=color, outline="gray")
    
    def on_click(self, event):
        x, y = event.y // 20, event.x // 20
        if self.mode.get() == "start":
            self.track.set_start(x, y)
        elif self.mode.get() == "finish":
            self.track.set_finish(x, y)
        elif self.mode.get() == "boundary":
            self.track.set_boundary(x, y)
        elif self.mode.get() == "visualize":
            if hasattr(self, 'visualized_cell'):
                prev_x, prev_y, prev_state = self.visualized_cell
                self.track.grid[prev_x, prev_y] = prev_state  # Restore previous state
            self.visualized_cell = (x, y, self.track.grid[x, y])  # Store current state
            self.track.grid[x, y] = 3  # Use 3 to represent the car
        self.draw_grid()
    
    def on_right_click(self, event):
        self.rect_start = (event.x, event.y)
        self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="blue")
    
    def on_right_drag(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], event.x, event.y)
    
    def on_right_release(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            x0, y0 = self.rect_start[1] // 20, self.rect_start[0] // 20
            x1, y1 = event.y // 20, event.x // 20
            x_range = sorted([x0, x1])
            y_range = sorted([y0, y1])
            self.track.set_boundaries_rectangle(x_range, y_range)
            self.draw_grid()
            self.rect_id = None
    
    def create_controls(self):
        start_button = tk.Radiobutton(self.left_frame, text="Set Start", variable=self.mode, value="start")
        start_button.grid(row=3, column=0, padx=5, pady=5)
        
        finish_button = tk.Radiobutton(self.left_frame, text="Set Finish", variable=self.mode, value="finish")
        finish_button.grid(row=3, column=1, padx=5, pady=5)
        
        boundary_button = tk.Radiobutton(self.left_frame, text="Set Boundary", variable=self.mode, value="boundary")
        boundary_button.grid(row=4, column=0, padx=5, pady=5)
        
        clear_button = tk.Button(self.left_frame, text="Clear Grid", command=self.clear_grid)
        clear_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        instructions_label = tk.Label(self.left_frame, text="Instructions:\n"
                                      "1. Set start, finish and boundary by ticking the adequate field\n" 
                                      "and then click the preferred cells. To select multiple cells\n"
                                      "at once to be the boundary you can hover over them\n"
                                      " while holding the right mouse button.\n"
                                      "2. The car can only drive right or forward,\n"
                                      "this should be reflected in the design of the track.\n"
                                      "3. Choose the algorithm parameters, using the default one is recommended.\n"
                                      "4. Press \"Start Learning\" and wait for the progress bar to fill upp.\n"
                                      "The first iteration can take couple of seconds but then the progress will speed up.\n"
                                      "5. Press \"Visualize Result\" to see how the car will drive starting from a\n"
                                      "random place on the start line.")
        instructions_label.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Label(self.right_frame, text="n_iter:").grid(row=0, column=0, padx=5, pady=5)
        self.n_iter_entry = tk.Entry(self.right_frame)
        self.n_iter_entry.insert(0, "10000")
        self.n_iter_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.right_frame, text="Initial Epsilon:").grid(row=1, column=0, padx=5, pady=5)
        self.initial_eps_entry = tk.Entry(self.right_frame)
        self.initial_eps_entry.insert(0, "0.5")
        self.initial_eps_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.right_frame, text="Subsequent Epsilon:").grid(row=2, column=0, padx=5, pady=5)
        self.subsequent_eps_entry = tk.Entry(self.right_frame)
        self.subsequent_eps_entry.insert(0, "0.01")
        self.subsequent_eps_entry.grid(row=2, column=1, padx=5, pady=5)
        
        start_learning_button = tk.Button(self.right_frame, text="Start Learning", command=self.start_learning)
        start_learning_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(self.right_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.visualize_button = tk.Button(self.right_frame, text="Visualize Result", command=self.start_visualization)
        self.visualize_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.visualize_button.config(state=tk.DISABLED)  # Initially disabled
    
    def clear_grid(self):
        if self.track:
            self.track.clear()
            self.draw_grid()
    
    def start_learning(self):
        def run_learning():
            n_iter = int(self.n_iter_entry.get())
            initial_eps = float(self.initial_eps_entry.get())
            subsequent_eps = float(self.subsequent_eps_entry.get())
            
            self.progress["maximum"] = n_iter
            self.progress["value"] = 0
            
            def progress_callback(iteration):
                self.progress["value"] = iteration
                self.root.update_idletasks()
            
            self.mc_control = OffPolicyMCControl(self.track, n_iter, initial_eps, subsequent_eps)
            self.mc_control.train(progress_callback)
            
            self.visualize_button.config(state=tk.NORMAL)
        
        # Run the learning process in a separate thread
        learning_thread = threading.Thread(target=run_learning)
        learning_thread.start()


    def start_visualization(self):
        if hasattr(self, 'visualized_cell'):
            prev_x, prev_y, prev_state = self.visualized_cell
            self.track.grid[prev_x, prev_y] = prev_state
            del self.visualized_cell

        episodes = self.mc_control.t_policy.generate_episode()
        self.positions = [e[0:2] for e in episodes] 
        self.current_position_index = 0
        self.animate_car()

    def animate_car(self):
        if self.current_position_index < len(self.positions):
            x, y = self.positions[self.current_position_index]
            
            if hasattr(self, 'visualized_cell'):
                prev_x, prev_y, prev_state = self.visualized_cell
                self.track.grid[prev_x, prev_y] = prev_state
            
            self.visualized_cell = (x, y, self.track.grid[x, y])
            self.track.grid[x, y] = 3 
            self.draw_grid()
            
            self.current_position_index += 1
            self.root.after(500, self.animate_car)
