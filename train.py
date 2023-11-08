import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import skimage
from SIREN.models.DARTS import DARTS
from SIREN.models.MLP import MLP
from SIREN.models.SIREN import SirenModel


def add_to_plot(axes, model_output, model_index, model_name):
    axes[model_index].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
    axes[model_index].set_title(model_name, fontsize=13)


def add_psnr_to_plot(axes, axis_index, psnr, model_name):
    axes[axis_index].plot(psnr, label=model_name, c='C0' if (i == 0) else 'mediumseagreen')
    axes[axis_index].set_xlabel('Iterations', fontsize=14)
    axes[axis_index].set_ylabel('PSNR', fontsize=14)
    axes[axis_index].legend(fontsize=13)


class TrainModelClass:
    def __init__(self, model, pixel_coordinates, loss=nn.MSELoss()):
        self.optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
        self.loss = loss
        self.model = model
        self.name = model.name
        self.pixel_coordinates = pixel_coordinates
        self.psnr = []

    def train(self, epochs=15000):
        for _ in tqdm(range(epochs)):
            model_output = self.forward()
            loss = self.loss(model_output, pixel_values)
            self.psnr.append(20 * np.log10(1.0 / np.sqrt(loss.cpu().detach())))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self):
        return self.model(self.pixel_coordinates)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

siren = SirenModel().to(device)
mlp = MLP().to(device)
darts = DARTS().to(device)

img = torch.from_numpy(skimage.data.camera())

# Target
img = ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5)
pixel_values = img.reshape(-1, 1).to(device)

# Input
resolution = img.shape[0]
pixel_linspace = torch.linspace(-1, 1, steps=resolution)
x, y = torch.meshgrid(pixel_linspace, pixel_linspace)
pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

train_model_list = [TrainModelClass(mlp, pixel_coordinates), TrainModelClass(siren, pixel_coordinates),
                    TrainModelClass(darts, pixel_coordinates)]

model_list = [darts, mlp, siren]
num_of_axis = len(model_list)

fig, axes = plt.subplots(1, num_of_axis + 2, figsize=(15, 3))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Ground Truth', fontsize=13)

img = img.to(device)
tau = 1.0

for i, train_model in enumerate(train_model_list):
    train_model.train(100)

    add_to_plot(axes, train_model.forward(), i + 1, train_model.name)
    add_psnr_to_plot(axes, len(train_model_list)-1, train_model.psnr, train_model.name)

for i in range(num_of_axis - 1):
    axes[i].set_xticks([])
    axes[i].set_yticks([])
axes[3].axis('off')
plt.savefig('Imgs/Siren.png')
plt.close()
