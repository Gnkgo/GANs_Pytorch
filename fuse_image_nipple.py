import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load images
original_image_path = 'result.png'
small_image_path = 'powerpoint_image.png'

original_image = Image.open(original_image_path).convert("RGBA")
small_image = Image.open(small_image_path).convert("RGBA")

# Function to create a sigmoid mask
def sigmoid(x, a=10):
    return 1 / (1 + np.exp(-a * (x - 0.5)))

def create_inverse_sigmoid_mask(size, transition_width=10):
    width, height = size
    mask = Image.new('L', (width, height), 255)
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - width/2)**2 + (y - height/2)**2) / (np.sqrt((width/2)**2 + (height/2)**2))
            mask.putpixel((x, y), int((1 - sigmoid(dist, transition_width)) * 255))
    return mask

# Function to blend images
def blend_images(original, small, positions, transition_width=10):
    blended_image = original.copy()
    mask = create_inverse_sigmoid_mask(small.size, transition_width)
    
    small_with_alpha = Image.new('RGBA', small.size)
    small_with_alpha.paste(small, (0, 0), small)
    small_with_alpha.putalpha(mask)
    
    small_width, small_height = small.size
    for position in positions:
        x, y = position
        paste_position = (x - small_width // 2, y - small_height // 2)
        blended_image.paste(small_with_alpha, paste_position, small_with_alpha)
    
    return blended_image

# Interactive function to get click positions and drag and drop
class InteractivePlacement:
    def __init__(self, image, num_positions):
        self.image = image
        self.num_positions = num_positions
        self.positions = []
        self.points = []
        self.dragging = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        
        # Create OK button
        self.ok_button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.ok_button = Button(self.ok_button_ax, 'OK')
        self.ok_button.on_clicked(self.on_ok)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        plt.show()

    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1 and len(self.positions) < self.num_positions:  # Left mouse button
            self.positions.append((int(event.xdata), int(event.ydata)))
            point = self.ax.plot(event.xdata, event.ydata, 'ro')[0]
            self.points.append(point)
            self.fig.canvas.draw()

    def on_drag(self, event):
        if self.dragging and len(self.positions) > 0:
            self.positions[-1] = (int(event.xdata), int(event.ydata))
            self.points[-1].set_data(event.xdata, event.ydata)
            self.fig.canvas.draw()

    def on_press(self, event):
        if event.inaxes == self.ax and event.button == 1 and len(self.positions) > 0:
            self.dragging = True

    def on_release(self, event):
        if event.button == 1:
            self.dragging = False

    def on_ok(self, event):
        plt.close(self.fig)
    
    def get_positions(self):
        return self.positions

# Get user input for number of placements
num_placements = int(input("Enter the number of placements (1 or 2): "))

# Validate input
if num_placements not in [1, 2]:
    raise ValueError("Number of placements must be 1 or 2.")

# Get click positions from user
interactive_placement = InteractivePlacement(original_image, num_placements)
click_positions = interactive_placement.get_positions()

# Blend the images
result_image = blend_images(original_image, small_image, click_positions)

# Save and show the result
result_image.save('blended_image3.png')
result_image.show()
