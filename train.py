import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TrainModelClass:
    def __init__(self, model, pixel_coordinates, pixel_values, loss=nn.MSELoss()):
        self.optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
        self.loss = loss
        self.model = model
        self.name = model.name
        self.pixel_coordinates = pixel_coordinates
        self.pixel_values = pixel_values
        self.psnr = []

    def train(self, epochs=15000):
        for _ in tqdm(range(epochs)):
            model_output = self.forward()
            loss = self.loss(model_output, self.pixel_values)
            self.psnr.append(20 * np.log10(1.0 / np.sqrt(loss.cpu().detach())))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self):
        return self.model(self.pixel_coordinates)
