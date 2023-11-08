import torch
from SIREN.data_manager import get_image, generate_pixel_values, generate_pixel_coordinates
from SIREN.models.MLP import MLP
from SIREN.models.SIREN import SirenModel
from SIREN.plotting import PlotManager
from SIREN.train import TrainModelClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

siren = SirenModel().to(device)
mlp = MLP().to(device)

img = get_image(device)
pixel_values = generate_pixel_values(device)
pixel_coordinates = generate_pixel_coordinates(device)


train_model_list = [TrainModelClass(mlp, pixel_coordinates, pixel_values),
                    TrainModelClass(siren, pixel_coordinates, pixel_values)]

num_of_axis = len(train_model_list)

plot_manager = PlotManager(num_of_axis, img.cpu())

for i, train_model in enumerate(train_model_list):
    train_model.train(10000)

    plot_manager.add_to_plot(train_model.forward(), i+1, train_model.name, img.shape[0])
    plot_manager.add_psnr_to_plot(len(train_model_list)-1, train_model.psnr, train_model.name)

plot_manager.finish_axis()
plot_manager.save()
