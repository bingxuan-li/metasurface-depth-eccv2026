import torch
import torch.nn.functional as f
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.linear_model as lm
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss
import warnings
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import roots_hermite
from PIL import Image
import h5py, cv2
from . import _occlusion as occ
import time


# First, we define a class for the imaging system layout. The origin of z axis is at the image plane. The origin of x and y axis is at the center of the lens.
class IncoherentLayout:
    def __init__(
        self,
        resolution,
        lens_z,
        lens_length,
        image_length,
        datatype=torch.float64,
        ifflip=False,
    ):  # ifflip is used to flip right and left
        # assert lens_length == detector_length*n_detector, "lens_length must be equal to detector_length*n_detector"
        self.resolution = (
            resolution  # resolution of the simulation, pixel number per micron
        )
        self.lens_z = lens_z  # distance from the lens to the detector plane, in microns
        self.lens_length = lens_length  # length of the lens, in microns
        self.image_length = image_length  # length of the image, in microns
        self.lens_size = round(
            self.lens_length * self.resolution
        )  # size of the lens in simulation pixels
        self.image_size = round(
            self.image_length * self.resolution
        )  # size of the image in simulation pixels
        self.fieldofview = (
            image_length / 2 / lens_z
        )  # field of view of the imaging system (tangent of half angle)

        self.datatype = datatype
        if datatype == torch.float64:
            self.complextype = torch.complex128
        elif datatype == torch.float32:
            self.complextype = torch.complex64
        else:
            raise ValueError("datatype must be torch.float64 or torch.float32")

        self.ifflip = ifflip


# Define the kernel for the propagation from lens to image plane
class Propagation:
    def __init__(
        self,
        layout: IncoherentLayout,
        wavelength_list: np.ndarray,
        weight_list: np.ndarray,
        xdisplacement=0,
        ydisplacement=0,
        device="cuda",
    ):
        self.datatype = layout.datatype
        self.complextype = layout.complextype
        self.layout = layout
        self.lens_size = layout.lens_size
        self.image_size = layout.image_size
        self.wavelength_list = wavelength_list
        self.weight_list = weight_list
        self.num_wavelength = len(self.wavelength_list)
        self.kernel_size = (
            layout.lens_size + layout.image_size - 1
        )  # 2*np.maximum(layout.lens_size,layout.image_size)-1
        self.xdisplacement = xdisplacement
        self.ydisplacement = ydisplacement
        self.device = device
        assert len(self.wavelength_list) == len(
            self.weight_list
        ), "wavelength_list and weight_list must have the same length"
        self.kernel_fftshift_tensor = torch.zeros(
            self.num_wavelength,
            1,
            self.kernel_size,
            self.kernel_size,
            dtype=self.complextype,
            device=device,
        )  # fft tensor of the shifted kernel
        # first dimension is wavelength, second dimension is point source, third and fourth dimension are the size of kernel
        self.get_kernelfft_tensor()

    def get_kernelfft_tensor(self):
        """
        get the kernel and fft of kernel for the propagation from lens to image plane
        """
        for i in range(len(self.wavelength_list)):
            kernel = self.get_kernel(self.wavelength_list[i])
            kernel_fftshift = torch.fft.fft2(torch.fft.ifftshift(kernel))
            self.kernel_fftshift_tensor[i, 0, :, :] = kernel_fftshift

        return self.kernel_fftshift_tensor

    def get_kernel(self, wavelength):
        """
        get the kernel for the propagation from lens to image plane
        """

        def diffraction_kernel_2d(x, y, z):
            r = torch.sqrt(x**2 + y**2 + z**2)
            w = (
                -(1j * 2 * torch.pi / wavelength - 1 / r)
                * z
                / (torch.pi * 2 * r * r)
                * torch.exp(1j * r * 2 * torch.pi / wavelength)
            )
            return w

        # get the grid of the kernel
        x = (
            torch.linspace(
                -self.kernel_size / 2 + 0.5,
                self.kernel_size / 2 - 0.5,
                steps=self.kernel_size,
                device=self.device,
                dtype=self.datatype,
            )
            / self.layout.resolution
        )
        y = x.detach().clone()
        x = x + self.xdisplacement
        y = y + self.ydisplacement
        coord_x, coord_y = torch.meshgrid(x, y, indexing="ij")
        G = diffraction_kernel_2d(coord_x, coord_y, self.layout.lens_z)
        return G.reshape((1,) + G.shape)

        # Then, we define the key functions for propagation

    def fourier_conv(self, signal: torch.Tensor) -> torch.Tensor:
        """
        args:
        signal, kernel: complex tensor, assume the images are square. the last 2 dim of signal is the height, and width of images.
        """
        s_size = signal.size()
        k_size = self.kernel_size
        padding = (k_size - s_size[-1]) // 2

        if (k_size - s_size[-1]) % 2 == 0:
            signal = f.pad(signal, (padding, padding, padding, padding))
        else:
            signal = f.pad(signal, (padding, padding + 1, padding, padding + 1))

        f_signal = torch.fft.fftn(signal, dim=(-2, -1))

        f_output = f_signal * self.kernel_fftshift_tensor
        f_output = torch.fft.ifftn(f_output, dim=(-2, -1))

        padding = (k_size - self.layout.image_size) // 2
        f_output = f_output[
            ...,
            padding : padding + self.layout.image_size,
            padding : padding + self.layout.image_size,
        ]

        return f_output

    def fourier_conv_batch(
        self, signal: torch.Tensor, wavelength_batch=3
    ) -> torch.Tensor:  # do the convolution for a batch of wavelengths
        """
        args:
        signal, kernel: complex tensor, assume the images are square. the dim of signal is (wavelength, sources, height, width)
        """
        s_size = signal.size()
        k_size = self.kernel_size
        padding = (k_size - s_size[-1]) // 2
        n_wavelength = s_size[0]

        if (k_size - s_size[-1]) % 2 == 0:
            signal = f.pad(signal, (padding, padding, padding, padding))
        else:
            signal = f.pad(signal, (padding, padding + 1, padding, padding + 1))

        for batch_num in range(0, n_wavelength, wavelength_batch):
            current_signal = signal[batch_num : batch_num + wavelength_batch]
            f_signal = torch.fft.fftn(current_signal, dim=(-2, -1))

            f_output = (
                f_signal
                * self.kernel_fftshift_tensor[batch_num : batch_num + wavelength_batch]
            )
            f_output = torch.fft.ifftn(f_output, dim=(-2, -1))

            padding = (k_size - self.layout.image_size) // 2
            f_output = f_output[
                ...,
                padding : padding + self.layout.image_size,
                padding : padding + self.layout.image_size,
            ]

            if batch_num == 0:
                output = f_output
            else:
                output = torch.cat((output, f_output), dim=0)

        return output


class ReadImageDepth:
    def __init__(
        self, rgb_file: str, depth_file: str, depth_magnification=1, ifshow=False
    ):
        self.depth_magnification = depth_magnification
        self.gray = self.read_image(rgb_file)
        self.depth = self.read_depth(depth_file) * depth_magnification
        self.d_min = torch.min(self.depth)
        self.d_max = torch.max(self.depth)
        self.d_scale = self.d_max / self.d_min
        self.height = self.gray.shape[0]
        self.width = self.gray.shape[1]
        assert self.gray.shape == self.depth.shape, "gray.shape != depth.shape"

        if ifshow:
            self.show_image()

    def read_image(self, rgb_file):
        image = Image.open(rgb_file)
        image = image.convert("L")
        image = np.array(image)
        if np.sum(image) < 1:
            raise ValueError(f"Image {rgb_file} is invalid.")
        image = torch.tensor(image, dtype=torch.float32)
        return image / 255.0  # Normalize to [0, 1]

    def read_depth(self, depth_file):
        if depth_file.endswith(".npy"):
            depth = np.load(depth_file)
        elif depth_file.endswith(".png") or depth_file.endswith(".jpg"):
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            if depth.ndim == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth = depth / 255.0  # Normalize to [0, 1]
        elif depth_file.endswith(".hdf5"):
            with h5py.File(depth_file, "r") as f:
                assert (
                    len(f.keys()) == 1
                ), "The depth hdf5 file should contain only one key"
                depth = f[list(f.keys())[0]][:]
        elif depth_file.endswith(".pfm"):
            depth = read_pfm(depth_file)
        # throw an error if the depth has nan/inf/negative values
        if np.isnan(depth).any() or np.isinf(depth).any() or (depth < 0).any():
            raise ValueError(
                f"Depth file {depth_file} contains invalid values: NaN, Inf, or negative values."
            )
        # depth: numpy.ndarray
        depth = torch.tensor(depth, dtype=torch.float32)
        return depth

    def show_image(self):
        plt.figure(figsize=(10, 20))

        # Plot gray image
        plt.subplot(2, 1, 1)
        plt.imshow(self.gray, cmap="gray")
        plt.title("Gray Image")
        plt.axis("off")  # Hide axis
        plt.colorbar()

        # Plot depth image
        plt.subplot(2, 1, 2)
        plt.imshow(self.depth, cmap="viridis")
        plt.title("Depth Image")
        plt.axis("off")  # Hide axis
        plt.colorbar()

        plt.tight_layout()
        plt.show()


# Finally, we define the incoherent optical neuron net work
class Metasurface(torch.nn.Module):
    def __init__(self, layout: IncoherentLayout, propagation: Propagation):
        super().__init__()
        self.layout = layout
        self.datatype = layout.datatype
        self.lens_size = layout.lens_size
        self.propagation = propagation
        ## Here we only use one quarter of the lens, and the phase is symmetric
        assert self.lens_size % 2 == 0, "lens_size must be even"
        self.phase = torch.nn.Parameter(
            torch.empty((1, 1, self.lens_size, self.lens_size), dtype=self.datatype)
        )  # the first dimension is for wavelength, the second dimension is for point source

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal * torch.exp(1j * self.phase)

    def initial_simplelens(self, wavelength, focus_x, focus_z, ifcircle=True):
        """
        initialize the phase of the metasurface for a simple lens
        """
        if focus_z == self.layout.lens_z:
            self.infocus_z = float("inf")
        else:
            self.infocus_z = (
                1 / (1 / focus_z - 1 / self.layout.lens_z) + self.layout.lens_z
            )
        self.focus_x = focus_x
        self.focus_z = focus_z

        k0 = 2 * torch.pi / wavelength
        # get the grid of the lens
        Xlens_array = (
            torch.linspace(
                -self.layout.lens_size / 2 + 0.5,
                self.layout.lens_size / 2 - 0.5,
                steps=self.layout.lens_size,
                dtype=self.datatype,
            )
            / self.layout.resolution
        )
        Ylens_array = Xlens_array
        Xlens_matrix, Ylens_matrix = torch.meshgrid(
            Xlens_array, Ylens_array, indexing="ij"
        )

        Focus1 = torch.tensor([focus_x, 0, focus_z]).to(self.datatype)

        # get the phase of the lens
        r1 = torch.sqrt(
            (Xlens_matrix - Focus1[0]) ** 2
            + (Ylens_matrix - Focus1[1]) ** 2
            + Focus1[2] ** 2
        ) - torch.sqrt(Focus1[0] ** 2 + Focus1[1] ** 2 + Focus1[2] ** 2)
        Lensphase = torch.angle(torch.exp(-1j * k0 * r1))
        if ifcircle:
            Lensphase[
                Xlens_matrix**2 + Ylens_matrix**2
                > (self.lens_size / self.layout.resolution / 2) ** 2
            ] = 0

        Lensphase = Lensphase.unsqueeze(0).unsqueeze(0)

        assert (
            Lensphase.shape == self.phase.shape
        ), "lensphase.shape != self.phase.shape"

        with torch.no_grad():
            self.phase.copy_(Lensphase)  # copy the phase to the metasurface

        return Lensphase

    def gen_pupilphase(self, L_number=10, epsilon=0.5, rotate180=False):

        # Create a pupil phase profile for rotating psf

        self.L_number = L_number
        self.epsilon = epsilon

        Xlens_array = (
            torch.linspace(
                -self.layout.lens_size / 2 + 0.5,
                self.layout.lens_size / 2 - 0.5,
                steps=self.layout.lens_size,
                dtype=self.datatype,
            )
            / self.layout.resolution
        )
        Ylens_array = Xlens_array
        Xlens_matrix, Ylens_matrix = torch.meshgrid(
            Xlens_array, Ylens_array, indexing="ij"
        )

        # get the pupil function
        Pupilphase = torch.zeros((self.lens_size, self.lens_size), dtype=self.datatype)
        r1 = torch.sqrt(Xlens_matrix**2 + Ylens_matrix**2) / (
            self.layout.lens_length / 2
        )
        angle = torch.angle(Xlens_matrix + 1j * Ylens_matrix)

        if rotate180:
            for l in range(L_number):
                select = (r1 >= (l / L_number) ** epsilon) & (
                    r1 < ((l + 1) / L_number) ** epsilon
                )
                Pupilphase[select] = (l + 1) * (angle[select] + torch.pi)
        else:
            for l in range(L_number):
                select = (r1 >= (l / L_number) ** epsilon) & (
                    r1 < ((l + 1) / L_number) ** epsilon
                )
                Pupilphase[select] = (l + 1) * angle[select]

        Pupilphase = Pupilphase.unsqueeze(0).unsqueeze(0)
        return Pupilphase

    def initial_pupillens(
        self, wavelength, focus_x, focus_z, L_number=10, epsilon=0.5, rotate180=False
    ):

        Pupilphase = self.gen_pupilphase(L_number, epsilon, rotate180)

        Pupilphase = Pupilphase.to(self.phase.device)
        if rotate180:
            focus_x = -focus_x
        self.initial_simplelens(wavelength, focus_x, focus_z)
        with torch.no_grad():
            self.phase.copy_(self.phase + Pupilphase)

    def set_phase(self, phase: torch.Tensor):
        """
        set the phase of the metasurface
        """
        assert phase.shape == self.phase.shape, "phase.shape != self.phase.shape"
        with torch.no_grad():
            self.phase.copy_(phase)

    def rotate90(self):
        """
        rotate the phase of the metasurface by 90 degrees
        """
        with torch.no_grad():
            self.phase.copy_(torch.rot90(self.phase, 1, (2, 3)))

    def planewave_pattern(self):
        field = torch.ones(
            (1, 1, self.lens_size, self.lens_size), dtype=torch.complex128
        ).cuda()
        field = self.forward(field)
        field = self.propagation.fourier_conv(field) / self.layout.resolution**2
        power = (torch.abs(field) ** 2) * (
            torch.tensor(self.propagation.weight_list, dtype=torch.float64).view(
                -1, 1, 1, 1
            )
        ).cuda()
        power = (power.squeeze(1)).sum(dim=0)

        if self.layout.ifflip:  # x->-x
            power = torch.flip(power, [0])

        Xmin = -self.layout.image_length / 2
        Xmax = self.layout.image_length / 2
        Ymin = -self.layout.image_length / 2
        Ymax = self.layout.image_length / 2

        fig, ax = plt.subplots()

        im = ax.imshow(
            power.transpose(0, 1).detach().to("cpu").numpy(),
            origin="lower",
            extent=(Xmin, Xmax, Ymin, Ymax),
        )

        fig.colorbar(im, ax=ax)

        ax.set_xlabel("X(μm)")
        ax.set_ylabel("Y(μm)")

        plt.show()

        return power

    def pointsource_test(self, pointsource):
        device0 = self.phase.device
        lens_z = self.layout.lens_z
        lens_size = self.layout.lens_size
        resolution = self.layout.resolution
        k0 = 2 * np.pi / self.propagation.wavelength_list
        k0 = torch.tensor(k0, dtype=self.datatype, device=device0)
        k0 = k0.view(-1, 1, 1, 1)
        deltax_array = (
            torch.linspace(
                -lens_size / 2 + 0.5,
                lens_size / 2 - 0.5,
                steps=lens_size,
                dtype=self.datatype,
                device=device0,
            )
            / resolution
        )
        deltay_array = deltax_array
        deltax_array = deltax_array - pointsource[0]
        deltay_array = deltay_array - pointsource[1]
        deltax_matrix = torch.zeros(
            lens_size, lens_size, dtype=self.datatype, device=device0
        )
        deltay_matrix = torch.zeros(
            lens_size, lens_size, dtype=self.datatype, device=device0
        )
        deltax_matrix[:, :], deltay_matrix[:, :] = torch.meshgrid(
            deltax_array, deltay_array, indexing="ij"
        )
        deltaz_matrix = pointsource[2]
        r_om = torch.sqrt(deltaz_matrix**2 + deltax_matrix**2 + deltay_matrix**2)
        r_om = r_om.unsqueeze(0)
        field = torch.exp(1j * k0 * r_om) / r_om
        ### Calculate the total incident power
        inc_power = (torch.abs(field) ** 2).sum() / resolution**2
        print("The total incident power is", inc_power)

        field = self.forward(field)
        field = (
            self.propagation.fourier_conv_batch(field, wavelength_batch=1)
            / self.layout.resolution**2
        )
        power = (torch.abs(field) ** 2) * (
            torch.tensor(self.propagation.weight_list, dtype=torch.float32).view(
                -1, 1, 1, 1
            )
        ).cuda()

        power = (power.squeeze(1)).sum(dim=0)

        ### Calculate the total output power
        out_power = power.sum() / resolution**2
        print("The total output power is", out_power)

        power = power / torch.max(power)  # normalize the image
        power = power.detach().to("cpu").numpy()

        if self.layout.ifflip:
            power = np.flip(power, [0])

        Xmin = -self.layout.image_length / 2
        Xmax = self.layout.image_length / 2
        Ymin = -self.layout.image_length / 2
        Ymax = self.layout.image_length / 2

        fig, ax = plt.subplots()

        im = ax.imshow(power.T, origin="lower", extent=(Xmin, Xmax, Ymin, Ymax))

        fig.colorbar(im, ax=ax)

        ax.set_xlabel("X(μm)")
        ax.set_ylabel("Y(μm)")

        plt.show()

        return power

    def gen_psflist(
        self,
        pointsource_list: torch.Tensor,
        downsample=6,
        pixelnum=63,
        batchsize=1,
        ifshift=True,
    ):
        device0 = self.phase.device
        lens_size = self.layout.lens_size
        image_size = self.layout.image_size
        resolution = self.layout.resolution
        k0 = 2 * np.pi / self.propagation.wavelength_list
        k0 = torch.tensor(k0, dtype=self.datatype, device=device0)
        k0 = k0.view(-1, 1, 1, 1)
        weights = torch.tensor(
            self.propagation.weight_list, dtype=self.datatype, device=device0
        )
        pointsource_list = pointsource_list.to(device0)
        n_point = len(pointsource_list)
        subsize = downsample * pixelnum
        self.psflist = torch.zeros(
            n_point, pixelnum, pixelnum, dtype=self.datatype, device=device0
        )
        self.shift = torch.zeros(
            n_point, 2, dtype=self.datatype, device=device0
        )  # x and y shift of the psf
        deltax_array = (
            torch.linspace(
                -lens_size / 2 + 0.5,
                lens_size / 2 - 0.5,
                steps=lens_size,
                dtype=self.datatype,
                device=device0,
            )
            / resolution
        )
        deltay_array = deltax_array
        deltax_matrix0, deltay_matrix0 = torch.meshgrid(
            deltax_array, deltay_array, indexing="ij"
        )
        for batch_num in range(0, n_point, batchsize):
            current_point = pointsource_list[batch_num : batch_num + batchsize]
            current_number = current_point.shape[0]
            r_om = torch.sqrt(
                deltax_matrix0.unsqueeze(0) ** 2
                + deltay_matrix0.unsqueeze(0) ** 2
                + current_point.unsqueeze(1).unsqueeze(1) ** 2
            )
            r_om = r_om.unsqueeze(0)
            field = torch.exp(1j * k0 * r_om)
            # From incident field to transmitted field
            field = self.forward(field)
            # From lens to image
            field = (
                self.propagation.fourier_conv_batch(field, wavelength_batch=1)
                / resolution**2
            )
            currentimage = torch.abs(field) ** 2 * weights.view(-1, 1, 1, 1)
            currentimage = currentimage.sum(dim=0)
            if self.layout.ifflip:
                currentimage = torch.flip(currentimage, [1])
            currentimage = torch.flip(
                currentimage, [2]
            )  # Default y axis is from top to bottom
            currentimage = currentimage.transpose(
                1, 2
            )  # now is y, x. Col index is x, row index is y
            currentimage = currentimage[
                :,
                image_size // 2 - subsize // 2 : image_size // 2 + subsize // 2,
                image_size // 2 - subsize // 2 : image_size // 2 + subsize // 2,
            ]
            if ifshift:
                self.shift[batch_num : batch_num + current_number, :] = (
                    self.comput_brightness_centroids2(currentimage, 51)
                    - (subsize - 1) / 2
                ) / downsample
            currentimage = currentimage.unfold(1, downsample, downsample).unfold(
                2, downsample, downsample
            )
            currentimage = (
                currentimage.contiguous()
                .view(current_number, -1, downsample, downsample)
                .sum(dim=(2, 3))
            )
            currentimage = currentimage.view(current_number, pixelnum, pixelnum)
            currentimage = currentimage / (downsample**2)
            self.psflist[batch_num : batch_num + current_number, :, :] = currentimage
        self.psflist = self.psflist.to(
            torch.float32
        )  # Ensure psflist is in the correct datatype
        return self.psflist, self.shift

    def comput_brightness_centroids(self, images: torch.Tensor, pixelnum: int):
        N, H, W = images.shape
        reshaped_images = images.reshape(N, -1)
        top_m_values, top_m_indices = torch.topk(reshaped_images, pixelnum, dim=1)
        rows = top_m_indices // W
        cols = top_m_indices % W
        total_brightness = top_m_values.sum(dim=1)
        centroid_rows = (rows.float() * top_m_values).sum(dim=1) / total_brightness
        centroid_cols = (cols.float() * top_m_values).sum(dim=1) / total_brightness
        centriods = torch.stack(
            (centroid_cols, centroid_rows), dim=1
        )  # col index is x, row index is y
        return centriods

    def comput_brightness_centroids2(self, images: torch.Tensor, pixelnum: int):
        assert pixelnum % 2 == 1, "pixelnum must be odd"
        N, H, W = images.shape
        reshaped_images = images.reshape(N, -1)
        flat_indices = reshaped_images.argmax(dim=1)
        rows = flat_indices // W
        cols = flat_indices % W
        offsets = torch.linspace(
            -pixelnum // 2,
            pixelnum // 2 + 1,
            pixelnum,
            dtype=torch.int,
            device=images.device,
        )
        row_offsets, col_offsets = torch.meshgrid(offsets, offsets, indexing="ij")
        rows_patch = rows[:, None, None] + row_offsets[None, :, :]
        cols_patch = cols[:, None, None] + col_offsets[None, :, :]
        rows_patch = rows_patch.clamp(0, H - 1)
        cols_patch = cols_patch.clamp(0, W - 1)
        batch_indices = torch.arange(N, device=images.device, dtype=torch.int)[
            :, None, None
        ]
        batch_indices = batch_indices.expand(N, pixelnum, pixelnum)
        patches = images[batch_indices, rows_patch, cols_patch]
        rows_patch = rows_patch.float()
        cols_patch = cols_patch.float()
        total_brightness = patches.sum(dim=(1, 2))
        centroid_rows = (rows_patch * patches).sum(dim=(1, 2)) / total_brightness
        centroid_cols = (cols_patch * patches).sum(dim=(1, 2)) / total_brightness
        centroids = torch.stack(
            (centroid_cols, centroid_rows), dim=1
        )  # col index is x, row index is y
        return centroids

    def show_phase(self):
        """
        show the phase of the metasurface
        """

        Xmin = -self.layout.lens_length / 2
        Xmax = self.layout.lens_length / 2
        Ymin = -self.layout.lens_length / 2
        Ymax = self.layout.lens_length / 2

        fig, ax = plt.subplots()
        phase = torch.remainder(self.phase, 2 * torch.pi)
        im = ax.imshow(
            phase.squeeze(0).squeeze(0).transpose(0, 1).detach().to("cpu").numpy(),
            cmap=cm.hsv,
            extent=(Xmin, Xmax, Ymin, Ymax),
            origin="lower",
            vmin=0,
            vmax=2 * np.pi,
            interpolation="None",
        )

        fig.colorbar(im, ax=ax)

        ax.set_xlabel("X(μm)")
        ax.set_ylabel("Y(μm)")

        plt.show()


class DepthEncoder(torch.nn.Module):
    def __init__(
        self,
        metasurface: Metasurface = None,
        depth_list: torch.Tensor = None,
        depth_sigma=5e3,
        max_memory=20,
        requires_grad=False,
        shotnoise=False,
        extension: int = 20,
        depth_crack: int = 10,
    ):
        super().__init__()
        if metasurface is not None:
            self.metasurface = metasurface
            self.metasurface.phase.requires_grad = requires_grad
        else:
            self.metasurface = None
        if depth_list is not None:
            self.depth_list = depth_list
            self.n_depth = len(depth_list)
        else:
            self.depth_list = None
            self.n_depth = 0
        self.max_memory = max_memory
        self.requires_grad = requires_grad
        self.shotnoise = shotnoise
        self.ph_num = 1e4
        self.depth_sigma = depth_sigma
        self.extension = extension
        self.depth_crack = depth_crack

    def gen_psflist(self, downsample=6, pixelnum=61, batchsize=1, ifshift=True):
        # assert pixelnum % 2 == 1, "pixelnum must be odd"
        self.psf_size = pixelnum
        self.metasurface.phase.requires_grad = self.requires_grad
        self.metasurface.gen_psflist(
            self.depth_list, downsample, pixelnum, batchsize, ifshift=ifshift
        )
        self.metasurface.propagation.kernel_fftshift_tensor = (
            self.metasurface.propagation.kernel_fftshift_tensor.cpu()
        )  # release GPU memory
        self.metasurface = self.metasurface.to(
            "cpu"
        )  # Move the metasurface to CPU to save GPU memory
        self.original_psflist = self.metasurface.psflist
        self.shift = self.metasurface.shift
        # get a psf with 180 degree rotation
        self.rotate_psflist = self.metasurface.psflist.clone()
        self.rotate_psflist = self.rotate_psflist.rot90(2, (1, 2))
        self.original_psflist = self.original_psflist / self.original_psflist.sum(
            dim=(1, 2)
        ).unsqueeze(1).unsqueeze(2)
        self.rotate_psflist = self.rotate_psflist / self.rotate_psflist.sum(
            dim=(1, 2)
        ).unsqueeze(1).unsqueeze(2)

    def save_psflist(self, filename: str):  # save to avoid using gen_psflist again
        torch.save(
            {
                "depth_list": self.depth_list,
                "original_psflist": self.original_psflist,
                "rotate_psflist": self.rotate_psflist,
                "psf_size": self.psf_size,
                "shift": self.shift,
            },
            filename,
        )

    def read_psflist(self, filename: str):
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        self.depth_list = checkpoint["depth_list"]
        self.n_depth = len(self.depth_list)
        self.original_psflist = checkpoint["original_psflist"]
        self.rotate_psflist = checkpoint["rotate_psflist"]
        self.psf_size = checkpoint["psf_size"]
        self.shift = checkpoint["shift"]
        # if there is metasurface, move it to cpu to save memory
        if self.metasurface is not None:
            self.metasurface.propagation.kernel_fftshift_tensor = (
                self.metasurface.propagation.kernel_fftshift_tensor.cpu()
            )  # release GPU memory
            self.metasurface = self.metasurface.to("cpu")

    def gen_disparity(self):
        self.disparity = torch.zeros(
            self.n_depth, 2, dtype=torch.float32, device=self.shift.device
        )  # angle and distance
        vector_tensor = self.shift
        self.disparity[:, 0] = torch.atan2(-vector_tensor[:, 1], vector_tensor[:, 0])
        self.disparity[:, 1] = torch.sqrt(
            vector_tensor[:, 0] ** 2 + vector_tensor[:, 1] ** 2
        )
        return self.depth_list, vector_tensor, self.disparity

    def forward(
        self,
        data: ReadImageDepth,
        device="cpu",
        ifshow=False,
        ifdisparity=False,
        ifhard=False,
        ifalpha=False,
    ):
        chunk_size = int(64 * self.max_memory / 24)
        batch_size = int(300 * self.max_memory / 24)
        self.original_psflist = self.original_psflist.to(device)
        self.rotate_psflist = self.rotate_psflist.to(device)
        self.depth_list = self.depth_list.to(device)

        H = data.height
        W = data.width
        E = self.extension
        gray_cpu = data.gray.cpu().numpy()
        depth_cpu = data.depth.cpu().numpy()

        output_tensor = torch.zeros(2, H, W, dtype=torch.float32, device=device)
        if ifhard:
            output_tensor[0, :, :] = self.outputimage_hard_chunked(
                gray=data.gray,
                depth=data.depth,
                psflist=self.original_psflist,
                device=device,
                chunk_size=chunk_size,
            )
            output_tensor[1, :, :] = self.outputimage_hard_chunked(
                gray=data.gray,
                depth=data.depth,
                psflist=self.rotate_psflist,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            # Prepare the weighted input image based on depth and padding
            data.gray  = torch.tensor(np.pad(data.gray.numpy(), ((E, E), (E, E)), mode='edge'))
            data.depth = torch.tensor(np.pad(data.depth.numpy(), ((E, E), (E, E)), mode='edge'))
            depthweights = self.get_depthweights(
                data.depth, device=device
            )  # (n_depth, H, W)
            depthweights = f.pad(
                depthweights, (0, self.psf_size - 1, 0, self.psf_size - 1)
            )  # Pad the depth weights to match PSF size
            # Create the extended input image
            # create a dictionary for extension
            data.gray = data.gray.to(device)
            data.depth = data.depth.to(device)
            output1 = self.outputimage(
                input=data,
                depthweights=depthweights,
                psf=self.original_psflist,
                device=device,
                batch_size=batch_size,
            )
            output2 = self.outputimage(
                input=data,
                depthweights=depthweights,
                psf=self.rotate_psflist,
                device=device,
                batch_size=batch_size,
            )

            output_tensor[0, :, :]= output1[0][E:-E, E:-E]
            output_tensor[1, :, :]= output2[0][E:-E, E:-E]
            alpha1 = output1[1][E:-E, E:-E]
            alpha2 = output2[1][E:-E, E:-E]

            # Now handle the extended regions
            ext_mask, ext_gray, ext_depth = occ.extend_edge_and_boundaries(
                gray=gray_cpu,
                depth=depth_cpu,
                n=E,
            )
            ext_mask = torch.from_numpy(ext_mask).to(device)
            ext_gray = torch.from_numpy(ext_gray).to(device).float()
            ext_depth = torch.from_numpy(ext_depth).to(device).float()
            ext_depth[ext_depth.isnan()] = 0

            ext_output1 = self.outputimage_hard_chunked(
                gray=ext_gray,
                depth=ext_depth,
                psflist=self.original_psflist,
                device=device,
                chunk_size=chunk_size,
            )
            ext_output2 = self.outputimage_hard_chunked(
                gray=ext_gray,
                depth=ext_depth,
                psflist=self.rotate_psflist,
                device=device,
                chunk_size=chunk_size,
            )
            ext_mask = ext_mask.float()
            ext_alpha1 = self.outputimage_hard_chunked(
                gray=ext_mask,
                depth=ext_depth,
                psflist=self.original_psflist,
                device=device,
                chunk_size=chunk_size,
            )
            ext_alpha2 = self.outputimage_hard_chunked(
                gray=ext_mask,
                depth=ext_depth,
                psflist=self.rotate_psflist,
                device=device,
                chunk_size=chunk_size,
            )

            ext_alpha1 = torch.clamp(ext_alpha1, max=1, min=0)
            ext_alpha2 = torch.clamp(ext_alpha2, max=1, min=0)
            output_tensor[0, :, :] = output_tensor[0, :, :] + ext_output1 * (1 - alpha1)
            output_tensor[1, :, :] = output_tensor[1, :, :] + ext_output2 * (1 - alpha2)
            alpha1 = alpha1 + (1 - alpha1) * ext_alpha1
            alpha2 = alpha2 + (1 - alpha2) * ext_alpha2
            # stack the alpha channel
            alpha_tensor = torch.stack([alpha1, alpha2], dim=0)

            for i in range(2):
                alpha = alpha_tensor[i, :, :]
                correction = torch.ones_like(alpha)
                correction = 1 / (alpha + 1e-6)
                alpha_tensor[i, :, :] = alpha * correction
                output_tensor[i, :, :] = output_tensor[i, :, :] * correction

        output_tensor[...] *= 255.0
        output_tensor[...] = output_tensor[...].clamp(0, 255)
        # if self.shotnoise:
        #     # Add shot noise to the output images
        #     times = self.ph_num / torch.max(output_tensor.view(2, -1), dim=1).values
        #     output_tensor = torch.poisson(
        #         output_tensor * times.view(-1, 1, 1)
        #     ) / times.view(-1, 1, 1)
        #     # ensure non-negative value
        #     output_tensor = torch.clamp(output_tensor, min=0)

        # normalize the output images to [0, 255]
        # output_tensor = output_tensor / torch.max(
        #     output_tensor.view(2, -1), dim=1
        # ).values.view(-1, 1, 1)
        # output_tensor = output_tensor * 255

        if ifshow:
            plt.figure(figsize=(10, 10))
            im = plt.imshow(data.gray.detach().to("cpu").numpy(), cmap="gray", vmin=0, vmax=1)
            plt.title("Gray Image")
            plt.axis("off")
            plt.colorbar(im, fraction=0.046, pad=0.04)  # Adjust spacing as needed
            plt.tight_layout()
            plt.show()

            for i in range(2):
                plt.figure(figsize=(10, 10))
                im = plt.imshow(
                    output_tensor[i, :, :].detach().to("cpu").numpy(),
                    cmap="gray",
                    vmax=255,
                    vmin=0,
                )
                plt.title("Output Image " + str(i))
                plt.axis("off")
                plt.colorbar(im, fraction=0.046, pad=0.04)  # Add colorbar to each
                plt.tight_layout()
                plt.show()

        if ifdisparity:
            disparity_tensor = self.disparity_image(data, device)
            return output_tensor, disparity_tensor

        if ifalpha:
            return output_tensor, alpha_tensor

        return output_tensor

    def disparity_image(self, input: ReadImageDepth, device="cpu"):
        H = input.height
        W = input.width
        N = H * W
        device1 = device
        depth_values = input.depth.to(device1).view(-1)  # (N,)
        dis_indices = torch.bucketize(depth_values, self.depth_list.to(device1))  # (N,)
        angle_tensor = self.disparity[dis_indices, 0].reshape(H, W)
        distance_tensor = self.disparity[dis_indices, 1].reshape(H, W)
        disparity_tensor = torch.stack((angle_tensor, distance_tensor), dim=0)
        return disparity_tensor

    def outputimage(
        self, input: ReadImageDepth, depthweights, psf, device="cpu", batch_size=25
    ):
        H1, W1 = input.height+self.extension*2, input.width+self.extension*2
        psf = psf.to(device)  # Ensure the PSF is on the correct device
        conv_image = torch.zeros(
            H1, W1, dtype=torch.float32, device=device
        )  # Initialize the output image tensor
        conv_alpha = torch.zeros(
            H1, W1, dtype=torch.float32, device=device
        )  # Initialize the alpha channel tensor
        occlusion = (
            torch.ones(H1, W1, dtype=torch.float32, device=device)
            * 2
            * self.n_depth
        )  # Initialize the occlusion tensor to record the pixel is occluded by which depth layer
        connection = torch.zeros(
            H1, W1, dtype=torch.float32, device=device
        )  # a buffer tensor which stores the connection of the current layer
        connection_alpha = torch.zeros(
            H1, W1, dtype=torch.float32, device=device
        )  # a buffer tensor which stores the connection alpha of the current layer
        gray = input.gray.to(device)
        gray = f.pad(
            gray, (0, self.psf_size - 1, 0, self.psf_size - 1)
        )  # Pad the input image to match PSF size
        pad = int((self.psf_size - 1) // 2)
        current_layer = 0
        occlusion_diff = (
            self.depth_crack
        )  # only when the depth difference is larger than occlusion_diff, we consider it is occluded
        for batch_num in range(0, self.n_depth, batch_size):
            current_psf = psf[batch_num : batch_num + batch_size]
            current_psf = f.pad(
                current_psf, (0, W1 - 1, 0, H1 - 1)
            )  # Pad the PSF to match the input image size
            f_psflist = torch.fft.rfft2(
                current_psf, dim=(-2, -1)
            )  # FFT of the current PSF
            current_image = torch.fft.rfft2(
                gray.unsqueeze(0) * depthweights[batch_num : batch_num + batch_size],
                dim=(-2, -1),
            )
            current_image = torch.fft.irfft2(current_image * f_psflist, dim=(-2, -1))
            current_image = current_image[
                :, pad : pad + H1, pad : pad + W1
            ]  # Crop to original image size
            current_alpha = torch.fft.rfft2(
                depthweights[batch_num : batch_num + batch_size], dim=(-2, -1)
            )
            current_alpha = torch.fft.irfft2(current_alpha * f_psflist, dim=(-2, -1))
            current_alpha = current_alpha[
                :, pad : pad + H1, pad : pad + W1
            ]
            for i in range(current_psf.shape[0]):
                occluded = occlusion < (current_layer - occlusion_diff)
                non_occluded = ~occluded
                conv_image[occluded] = conv_image[occluded] + connection[occluded] * (
                    1 - conv_alpha[occluded]
                )
                conv_alpha[occluded] = conv_alpha[occluded] + connection_alpha[occluded] * (
                    1 - conv_alpha[occluded]
                )
                connection[occluded] = 0
                connection_alpha[occluded] = 0
                connection = connection + current_image[i]
                connection_alpha = connection_alpha + current_alpha[i]
                current_mask = current_alpha[i] > 1e-2
                occlusion[current_mask] = current_layer
                current_layer += 1
        conv_image = conv_image + connection * (1 - conv_alpha)
        conv_alpha = conv_alpha + connection_alpha * (1 - conv_alpha)
        # correction = torch.ones_like(conv_image)
        # correction[conv_alpha > 0.95] = 1 / conv_alpha[conv_alpha > 0.95]
        # conv_image = conv_image * correction
        # conv_alpha = conv_alpha * correction

        return conv_image, conv_alpha

    def get_depthweights(self, depth: torch.Tensor, device="cpu"):
        """
        Compute per-depth-layer weights.
        Pixels with NaN depth are treated as 'non-participating':
        their weights across all layers will be 0.

        Args:
            depth (torch.Tensor): (H, W) depth map, where NaN marks non-participating pixels.
            device (str, optional): 'cpu' or 'cuda'. Defaults to depth.device if None.

        Returns:
            torch.Tensor: (n_depth, H, W) normalized depth-layer weights.
        """
        depth = depth.to(device).float()  # (H, W)
        H, W = depth.shape[-2], depth.shape[-1]

        # 1) Create a mask of valid (finite) pixels
        valid_mask = torch.isfinite(depth)  # (H, W) bool

        # 2) Replace invalid (NaN) pixels with 0 for computation
        #    These will be set to zero later using the mask
        depth_filled = torch.where(valid_mask, depth, torch.zeros_like(depth))

        # 3) Compute depth difference for each layer using broadcasting
        depth_list = self.depth_list.to(device).float()  # (n_depth,)
        depth_list = depth_list.view(self.n_depth, 1, 1)  # (n_depth,1,1)
        depthlayer = depth_filled.unsqueeze(0) - depth_list  # (n_depth,H,W)

        # 4) Apply Gaussian weighting (elementwise)
        weights = gaussian_weighting(
            depthlayer, sigma=self.depth_sigma, clip_limit=20
        ).to(
            torch.float32
        )  # (n_depth,H,W)

        # 5) Zero out all weights for invalid pixels
        weights = weights * valid_mask.unsqueeze(0)  # (n_depth,H,W)

        # 6) Normalize only where there are valid weights
        sums = weights.sum(dim=0, keepdim=True)  # (1,H,W)
        depthweights = torch.where(
            sums > 0, weights / (sums + 1e-8), torch.zeros_like(weights)
        )  # (n_depth,H,W)

        return depthweights

    def outputimage_hard_chunked(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
        psflist: torch.Tensor,
        device="cpu",
        chunk_size=64,
    ):
        H, W = gray.shape[-2], gray.shape[-1]
        offset = self.psf_size // 2
        num_psfs = psflist.shape[0]
        depth_list = self.depth_list.to(device)

        gray = gray.to(device)
        depth = depth.to(device)
        psflist = psflist.to(device)

        # 翻转 PSF 核
        psflist_flipped = torch.flip(psflist, dims=[1, 2])  # (num_psfs, k, k)

        depth_values = depth.view(-1)
        psf_indices = torch.bucketize(depth_values, depth_list)
        psf_indices = psf_indices.clamp(0, num_psfs - 1).view(H, W)

        # one-hot mask (H, W, num_psfs)
        one_hot = f.one_hot(psf_indices, num_classes=num_psfs).to(gray.dtype)
        output_image = torch.zeros((H, W), dtype=gray.dtype, device=device)

        # 分块处理
        for i in range(0, num_psfs, chunk_size):
            j = min(i + chunk_size, num_psfs)
            current_chunk = j - i

            psf_chunk = psflist_flipped[i:j]  # (chunk, k, k)
            one_hot_chunk = one_hot[..., i:j]  # (H, W, chunk)
            gray_maskeds = (
                (gray.unsqueeze(-1) * one_hot_chunk).permute(2, 0, 1).unsqueeze(0)
            )  # (1, chunk, H, W)
            psf_chunk = psf_chunk.unsqueeze(1)  # [chunk, 1, k, k]
            # 分组卷积
            blurred = f.conv2d(
                gray_maskeds, psf_chunk, padding=offset, groups=current_chunk
            )  # (1, chunk, H, W)
            output_image += torch.sum(blurred, dim=1).squeeze(0)

        return output_image

    def outputimage_hard_grouped(
        self, gray: torch.Tensor, depth: torch.Tensor, psflist: torch.Tensor, device="cpu"
    ):
        H, W = gray.shape[-2], gray.shape[-1]
        offset = self.psf_size // 2
        device1 = device
        gray = gray.to(device1)
        depth = depth.to(device1)
        psflist = psflist.to(device1)

        depth_values = depth.view(-1)
        psf_indices = torch.bucketize(depth_values, self.depth_list.to(device1))
        psf_indices = psf_indices.clamp(0, len(self.depth_list) - 1).view(H, W)

        output_image = torch.zeros(H, W, dtype=psflist.dtype, device=device1)

        for psf_id in range(len(self.depth_list)):
            mask = (psf_indices == psf_id).to(gray.dtype)  # 掩码，避免int-float不匹配
            gray_masked = gray * mask  # 只保留该PSF的像素
            psf = torch.flip(psflist[psf_id], dims=[0, 1])  # 旋转PSF以进行卷积
            psf = psf.unsqueeze(0).unsqueeze(0)  # (1,1,psf_size,psf_size)
            blurred = f.conv2d(gray_masked.unsqueeze(0).unsqueeze(0), psf, padding=offset)
            output_image += blurred[0, 0]

        return output_image

def GH_quadrature(center, FWHM, n):
    """
    center: center of the gaussian quadrature
    FHWM: full width at half maximum
    n: number of points
    """
    std_dev_wavelength = FWHM / (2 * np.sqrt(2 * np.log(2)))
    x, w = roots_hermite(n)
    sample_wavelength = center + std_dev_wavelength * np.sqrt(2) * x
    weights_wavelength = w / np.sum(w)
    return sample_wavelength, weights_wavelength


def GL_quadrature(wavelength, spectrum, n):
    """
    wavelength: the np.array of the wavelength
    spectrum: the np.array of the spectrum
    n: number of points
    """
    a, b = wavelength[0], wavelength[-1]
    cs_w = CubicSpline(wavelength, spectrum, bc_type="natural")
    xg_0to1, wg_0to1 = leggauss(n)  # x in [-1,1]
    # Remap to [a,b]
    xg = 0.5 * (b - a) * xg_0to1 + 0.5 * (b + a)  # 7 node positions
    wg = wg_0to1 * cs_w(xg)  # 7 node weights
    wg = wg / np.sum(wg)  # Normalize the weights
    return xg, wg


def gaussian_weighting(depth_diff, sigma=5e3, clip_limit=20):
    scaled_diff = depth_diff / sigma
    scaled_diff = torch.clamp(scaled_diff, -clip_limit, clip_limit)
    return torch.exp(-0.5 * scaled_diff**2)


def pad_2D(
    tensor: torch.Tensor, diff_x: int, diff_y: int, value: float = 0.0
) -> torch.Tensor:
    """
    Symmetrically pad a 2D (or 3D: [D,H,W]) tensor by (diff_y, diff_x).
    Ensures total output size = (H+diff_y, W+diff_x), center-aligned.

    Args:
        tensor: (H,W) or (D,H,W)
        diff_x: total pixels to add along width, the last dimension
        diff_y: total pixels to add along height, the second last dimension
        value:  fill value (default=0)

    Returns:
        padded tensor of shape (..., H+diff_y, W+diff_x)
    """
    pad_x_left = diff_x // 2
    pad_x_right = diff_x - pad_x_left
    pad_y_top = diff_y // 2
    pad_y_bottom = diff_y - pad_y_top

    return f.pad(
        tensor,
        (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom),
        mode="constant",
        value=value,
    )
