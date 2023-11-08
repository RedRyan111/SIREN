import skimage
import torch


def get_image(device):
    return ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5).to(device)


def generate_pixel_values(device):
    img = get_image(device)
    pixel_values = img.reshape(-1, 1)
    return pixel_values


def generate_pixel_coordinates(device):
    img = get_image(device)
    resolution = img.shape[0]
    pixel_linspace = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(pixel_linspace, pixel_linspace)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)
    return pixel_coordinates
