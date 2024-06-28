import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import numpy as np
import cv2
import os

# Sample image data
file_name = '146.jpg'
image_data = cv2.imread(file_name)
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

image_data = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)

class InteractiveRectangle:
    def __init__(self, image_data, num_rectangles):
        self.image_data = image_data
        self.num_rectangles = num_rectangles
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True)

        self.img_item = pg.ImageItem(image_data)
        self.view.addItem(self.img_item)
        
        self.rects = []
        self.rois = []
        self.rect_count = 0

        self.view.scene().sigMouseClicked.connect(self.add_roi)

        # Add OK button
        self.btn_ok = QtWidgets.QPushButton('OK')
        self.btn_ok.clicked.connect(self.save_coords)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.win)
        self.layout.addWidget(self.btn_ok)
        self.container = QtWidgets.QWidget()
        self.container.setLayout(self.layout)
        self.container.show()

        self.app.exec_()

    def add_roi(self, event):
        if self.rect_count < self.num_rectangles:
            pos = event.scenePos()
            mouse_point = self.view.mapSceneToView(pos)
            roi = pg.RectROI([mouse_point.x() - 10, mouse_point.y() - 10], [20, 20], pen=(0, 9))
            self.view.addItem(roi)
            roi.sigRegionChanged.connect(self.update_rects)
            self.rois.append(roi)
            self.rect_count += 1
            self.update_rects()  # Save coordinates immediately after creation

    def update_rects(self):
        self.rects = [(roi.pos(), roi.size()) for roi in self.rois]

    def get_rect_coords(self):
        return self.rects

    def save_coords(self):
        
        coords_list = self.get_rect_coords()
        print(coords_list)
        if coords_list:
            for idx, (pos, size) in enumerate(coords_list):
                coords = (pos.x(), pos.y(), size.x(), size.y())
                print(f"Rectangle {idx + 1} coordinates: {coords}")
                mask = self.create_mask(coords)
                name = os.path.basename(file_name).split('.')[0]
                self.save_mask(mask, name, idx + 1)
        else:
            print("No rectangles drawn.")
        self.app.quit()

    def create_mask(self, coords):
        mask = np.zeros(self.image_data.shape[:2], dtype=np.uint8)
        x0, y0 = int(coords[0]), int(coords[1])
        x1, y1 = x0 + int(coords[2]), y0 + int(coords[3])
        mask[y0:y1, x0:x1] = 1
        return mask

    def save_mask(self, mask, name, idx):
        mask_name = f'scar_{name}_mask{idx:03d}.png'
        cv2.imwrite(mask_name, mask * 255)
        print(f'Mask saved as {mask_name}')

# Create an instance of InteractiveRectangle
num_rectangles = 2  # or set to 1 if only one rectangle is needed
interactive_rectangle = InteractiveRectangle(image_data, num_rectangles)
