import torch
import torch.nn as nn
from torch.nn import Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh
import matplotlib.pyplot as plt
import numpy as np

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

model_path = 'model_epoch_699.pth'
state_dict = torch.load(model_path, map_location=device)
generator.load_state_dict(state_dict['generator_state_dict'])
generator.eval()

noise = torch.randn(1, 100, 1, 1, device=device)

with torch.no_grad():
    generated_image = generator(noise)

generated_image = generated_image.squeeze(0).detach().cpu().numpy()  
generated_image = np.transpose(generated_image, (1, 2, 0)) 

# plt.imshow(generated_image)
# plt.axis('off')  
# plt.show()
# Save the image
output_path = 'generated_image3.png'
plt.imsave(output_path, generated_image)