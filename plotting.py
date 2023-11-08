from matplotlib import pyplot as plt


class PlotManager:
    def __init__(self, num_of_axis, img):
        self.num_of_axis = num_of_axis + 2
        self.fig, self.axes = plt.subplots(1, self.num_of_axis, figsize=(15, 5))
        self.axes[0].imshow(img, cmap='gray')
        self.axes[0].set_title('Ground Truth', fontsize=13)

    def add_to_plot(self, pixel_values, axis_index, name, resolution):
        self.axes[axis_index].imshow(pixel_values.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
        self.axes[axis_index].set_title(name, fontsize=13)

    def add_psnr_to_plot(self, axis_index, psnr, name):
        self.axes[self.num_of_axis-1].plot(psnr, label=name, c='C0' if (axis_index == 0) else 'mediumseagreen')
        self.axes[self.num_of_axis-1].set_xlabel('Iterations', fontsize=14)
        self.axes[self.num_of_axis-1].set_ylabel('PSNR', fontsize=14)
        self.axes[self.num_of_axis-1].legend(fontsize=13)

    def finish_axis(self):
        for i in range(self.num_of_axis - 1):
            self.axes[i].set_xticks([])
            self.axes[i].set_yticks([])

    def save(self):
        plt.savefig('Imgs/Siren.png')
        plt.close()
