# pyright: strict

import math
import numpy as np
import matplotlib.pyplot as plt


class ForceGenerator2:
    def __init__(
        self,
        width: float,
        height: float,
        safety_margin: float,
        r_min: float,
        r_max: float,
        smoothness: float = 1.2,
        sphere_factor: float = 1.2,
        normal_peak: float = 125000,
        shear_peak: float = 2500,
        density: float = 0.1
    ):
        self.width = width
        self.height = height
        self.safety_margin = safety_margin
        self.r_min = r_min
        self.r_max = r_max
        self.smoothness = smoothness
        self.sphere_factor = sphere_factor
        self.normal_peak = normal_peak
        self.shear_peak = shear_peak

        self.x0, self.y0 = np.random.uniform(0.03, 0.07, 2)
        self.rx, self.ry = np.random.uniform(0.02, 0.03, 2)

        r_avg = (self.rx + self.ry) / 2.0
        volume = 4 / 3 * np.pi * self.rx * self.ry * r_avg

        self.g = (
            density * volume * 9.81 * np.random.uniform(0.95, 1.05)
            / (np.pi * self.rx * self.ry)
        )

    def __call__(self, x: float, y: float):
        x0, y0, rx, ry = self.x0, self.y0, self.rx, self.ry

        d_sq = ((x - x0) ** 2 / rx ** 2) + ((y - y0) ** 2 / ry ** 2)

        normal = math.sqrt(
            1 - math.pow(min(1, d_sq), self.sphere_factor)
        ) * self.normal_peak

        if normal > 0:
            normal *= 1 + np.random.normal(0, 0.05)

        if d_sq < 1:
            shear = (
                math.pow(d_sq, 0.3)
                + math.pow(1.0 - d_sq, 0.3)
            ) / math.pow(0.5, 0.6)

            dx, dy = x - x0, y - y0
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist == 0:
                shear_x = 0
                shear_y = 0
            else:
                shear_x = shear * (dx / dist)
                shear_y = shear * (dy / dist)

            shear_x *= self.shear_peak
            shear_y *= self.shear_peak

            shear_x *= 1 + np.random.normal(0, 0.05)
            shear_y *= 1 + np.random.normal(0, 0.05)

            shear_y += self.g
        else:
            shear_x = 0
            shear_y = 0

        return normal, shear_x, shear_y

    def show(self):
        resolution = 0.0025

        x_range = np.arange(0, self.width + resolution, resolution)
        y_range = np.arange(0, self.height + resolution, resolution)

        X, Y = np.meshgrid(x_range, y_range)

        normal_grid, shear_x_grid, shear_y_grid = np.vectorize(
            self,
            otypes=[float, float, float]
        )(X, Y)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        im1 = axes[0].pcolormesh(
            X,
            Y,
            normal_grid,
            cmap="viridis",
            shading="auto"
        )
        axes[0].set_title("Normale kracht ($N/m^2$)")
        axes[0].set_aspect("equal")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(
            X,
            Y,
            shear_x_grid,
            cmap="RdBu_r",
            shading="auto",
            vmin=np.min(shear_x_grid),
            vmax=np.max(shear_x_grid)
        )
        axes[1].set_title("Schuifkracht X-component")
        axes[1].set_aspect("equal")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(
            X,
            Y,
            shear_y_grid,
            cmap="RdBu_r",
            shading="auto",
            vmin=np.min(shear_y_grid),
            vmax=np.max(shear_y_grid)
        )
        axes[2].set_title(f"Schuifkracht Y-component\nIncl. {self.g:.1f} g")
        axes[2].set_aspect("equal")
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()


def main():
    peak_normal = 5 / (0.01 * 0.01)

    forceg = ForceGenerator2(
        0.1,
        0.1,
        0.03,
        0.01,
        0.02,
        normal_peak=peak_normal,
        shear_peak=peak_normal * 0.2
    )

    forceg.show()


if __name__ == "__main__":
    main()