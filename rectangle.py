import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button

# Sample image data
image_data = np.random.rand(10, 10)

class InteractiveRectangle:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.rect = None
        self.rect_coords = None
        self.rs = RectangleSelector(
            ax, self.on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.cid = self.image.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_select(self, eclick, erelease):
        if self.rect is not None:
            self.rect.set_visible(False)
        self.rect = plt.Rectangle(
            (eclick.xdata, eclick.ydata),
            erelease.xdata - eclick.xdata,
            erelease.ydata - eclick.ydata,
            edgecolor='red',
            fill=False
        )
        self.ax.add_patch(self.rect)
        self.image.figure.canvas.draw()
        self.rect_coords = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    def on_key_press(self, event):
        if event.key == 'r' and self.rect is not None:
            self.rect.set_visible(False)
            self.image.figure.canvas.draw()
            self.rect = None

    def get_rect_coords(self):
        return self.rect_coords

def save_coords(event):
    coords = interactive_rectangle.get_rect_coords()
    if coords is not None:
        print(f"Rectangle coordinates: {coords}")
    else:
        print("No rectangle drawn.")

fig, ax = plt.subplots()
ax.imshow(image_data, cmap='gray')

# Add OK button
ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
btn_ok = Button(ax_button, 'OK')
btn_ok.on_clicked(save_coords)

interactive_rectangle = InteractiveRectangle(ax, ax.images[0])

plt.show()
