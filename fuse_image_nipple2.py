import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load images
original_image_path = '146.jpg'
small_image_path = 'test.jpg'

original_image = Image.open(original_image_path).convert("RGBA")
small_image = Image.open(small_image_path).convert("RGBA")

# Function to create a sigmoid 
def sigmoid(x, a=10):
    return 1 / (1 + np.exp(-a * (x - 0.5)))

def create_inverse_sigmoid_mask(size, transition_width=10):
    width, height = size
    mask = np.zeros((height, width), dtype=np.float32)
    center_x, center_y = width / 2, height / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_radius
            mask[y, x] = 1 - sigmoid(dist, transition_width)
    
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask, mode='L')

# Function to adjust the color of the small image
def adjust_color(small, target_color):
    small = np.array(small, dtype=np.float32)
    small_color_mean = np.mean(small[..., :3], axis=(0, 1))
    
    for channel in range(3):
        small[..., channel] *= target_color[channel] / small_color_mean[channel]
    
    small = np.clip(small, 0, 255).astype(np.uint8)
    return Image.fromarray(small, 'RGBA')

# Function to calculate the average color of a region
def calculate_average_color(image, position, region_size):
    x, y = position
    x_start = max(x - region_size // 2, 0)
    y_start = max(y - region_size // 2, 0)
    x_end = min(x + region_size // 2, image.width)
    y_end = min(y + region_size // 2, image.height)
    
    region = np.array(image.crop((x_start, y_start, x_end, y_end)))
    avg_color = np.mean(region[..., :3], axis=(0, 1))
    return avg_color

# Function to resize the small image to 40x40 pixels
def resize_image(image):
    return image.resize((50, 50), Image.Resampling.LANCZOS)

def mirror_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# Function to blend images
def blend_images(original, small, positions, transition_width=10):
    original = np.array(original)
    blended_image = original.copy()

    if (len(positions) == 0):
        return Image.fromarray(blended_image.astype('uint8'), 'RGBA')
    if (len(positions) == 2):
        small = resize_image(small)  # Resize the small image to 40x40 pixels
    else:
        small = mirror_image(small)
    mask = create_inverse_sigmoid_mask(small.size, transition_width)
    mask = np.array(mask) / 255.0  # Normalize mask to [0, 1]

    small = np.array(small)
    
    small_height, small_width = small.shape[:2]
    for position in positions:
        avg_color = calculate_average_color(Image.fromarray(original), position, max(small_width, small_height))
        adjusted_small = np.array(adjust_color(Image.fromarray(small), avg_color))
        
        x, y = position
        x_start = max(x - small_width // 2, 0)
        y_start = max(y - small_height // 2, 0)
        x_end = min(x_start + small_width, blended_image.shape[1])
        y_end = min(y_start + small_height, blended_image.shape[0])
        
        x_start_small = max(0, - (x - small_width // 2))
        y_start_small = max(0, - (y - small_height // 2))
        x_end_small = x_start_small + (x_end - x_start)
        y_end_small = y_start_small + (y_end - y_start)

        alpha = mask[y_start_small:y_end_small, x_start_small:x_end_small, np.newaxis]
        blended_image[y_start:y_end, x_start:x_end] = (alpha * adjusted_small[y_start_small:y_end_small, x_start_small:x_end_small] + 
                                                      (1 - alpha) * blended_image[y_start:y_end, x_start:x_end])
    
    return Image.fromarray(blended_image.astype('uint8'), 'RGBA')

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
            self.points[-1].set_data([event.xdata], [event.ydata])  # Wrap in lists
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
num_placements = int(input("Enter the number of placements (0, 1 or 2): "))

# Validate input
if num_placements not in [0, 1, 2]:
    raise ValueError("Number of placements must be 0, 1 or 2.")

# Get click positions from user
if (num_placements > 0):
    interactive_placement = InteractivePlacement(original_image, num_placements)

    click_positions = interactive_placement.get_positions()

# Blend the images
    result_image = blend_images(original_image, small_image, click_positions)
else :
    result_image = Image.open('146.png')
# Save and show the result
result_image.save('blended_image4.png')
result_image.show()
