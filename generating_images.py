import torch
import torch.nn as nn
from torch.nn import Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh
import matplotlib.pyplot as plt
import numpy as np

# Define the generator architecture (must match the architecture used during training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = Sequential(
            ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            BatchNorm2d(512),
            ReLU(True),
            ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            ReLU(True),
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            ReLU(True),
            ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            Tanh()
        )

    def forward(self, input):
        return self.gen(input)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Adjust the map_location parameter to handle systems without CUDA
model_path = 'model_epoch_900.pth'
state_dict = torch.load(model_path, map_location=device)
generator.load_state_dict(state_dict['generator_state_dict'])
generator.eval()

# Generate random noise as input to the generator
noise = torch.randn(1, 100, 1, 1, device=device)

# Generate an image
with torch.no_grad():
    generated_image = generator(noise)

# Prepare the image for display
generated_image = generated_image.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and detach from the graph
generated_image = np.transpose(generated_image, (1, 2, 0))  # Reorder dimensions for image display
#generated_image = (generated_image + 1) / 2  # Rescale to [0,1] from [-1,1]

# Display the image
plt.imshow(generated_image)
plt.axis('off')  # Hide axis
plt.show()
