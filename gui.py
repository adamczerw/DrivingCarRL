import tkinter as tk

class GridApp:
    def __init__(self, root, rows, cols):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.buttons = []
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.canvas = tk.Canvas(root, width=cols*50, height=rows*50)
        self.canvas.grid(row=0, column=0, rowspan=rows, columnspan=cols)
        self.canvas.bind('<Button-1>', self.on_left_click)
        self.canvas.bind('<Button-3>', self.on_right_click)
        self.canvas.bind('<B3-Motion>', self.on_right_drag)
        self.canvas.bind('<ButtonRelease-3>', self.on_right_release)

        for row in range(rows):
            button_row = []
            for col in range(cols):
                button = tk.Button(root, width=4, height=2)
                button.grid(row=row, column=col)
                button_row.append(button)
            self.buttons.append(button_row)

    def on_left_click(self, event):
        row, col = self.get_grid_position(event.x, event.y)
        if row is not None and col is not None:
            self.change_color(row, col)

    def on_right_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='black')

    def on_right_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_right_release(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            self.color_cells_in_rectangle(self.start_x, self.start_y, event.x, event.y)

    def get_grid_position(self, x, y):
        row = y // 50
        col = x // 50
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None, None

    def change_color(self, row, col):
        current_color = self.buttons[row][col].cget('bg')
        new_color = 'red' if current_color == 'SystemButtonFace' else 'SystemButtonFace'
        self.buttons[row][col].configure(bg=new_color)

    def color_cells_in_rectangle(self, x1, y1, x2, y2):
        row1, col1 = self.get_grid_position(x1, y1)
        row2, col2 = self.get_grid_position(x2, y2)
        if row1 is not None and col1 is not None and row2 is not None and col2 is not None:
            for row in range(min(row1, row2), max(row1, row2) + 1):
                for col in range(min(col1, col2), max(col1, col2) + 1):
                    self.buttons[row][col].configure(bg='red')

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Grid Color App")
    app = GridApp(root, 10, 10)  # Change 10, 10 to n, m for different grid sizes
    root.mainloop()