from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def generate_normal_force(
    x,
    y,
    x0,
    y0,
    rx,
    ry,
    smoothness=1.0,
    sphere_factor=1.2,
    peak_val=1.0,
):
    d_sq = ((x - x0) ** 2 / rx ** 2) + ((y - y0) ** 2 / ry ** 2)
    d_sq = np.clip(d_sq, 0.0, 1.0)

    z_norm = np.sqrt(1.0 - d_sq ** sphere_factor)
    z_norm[d_sq >= 1.0] = 0.0

    noise = np.random.normal(0.0, 0.02, x.shape)
    z_norm = np.where(z_norm > 0.0, z_norm + noise, 0.0)

    z_norm_smooth = gaussian_filter(z_norm, sigma=smoothness)
    z_norm_smooth[d_sq > 0.95] = 0.0
    z_norm_smooth = np.maximum(z_norm_smooth, 0.0)

    return z_norm_smooth * peak_val, d_sq


def generate_shear_force(x, y, d_sq, peak_val):
    raw_profile = (d_sq ** 0.3) * ((1.0 - d_sq) ** 0.3)

    max_raw = np.max(raw_profile)

    if max_raw <= 0.0:
        max_raw = 1.0

    normalized_profile = raw_profile / max_raw

    shear_noise = np.random.normal(0.0, 0.01, x.shape)

    shear_val = np.where(d_sq < 1.0, (normalized_profile + shear_noise) * peak_val, 0.0,)

    return np.maximum(shear_val, 0.0)


class ForceGenerator2:

    def __init__(
        self,
        width=0.1,
        height=0.1,
        safety_margin=0.03,
        r_min=0.02,
        r_max=0.03,
        smoothness=1.2,
        sphere_factor=1.2,
        normal_peak=125000.0,
        shear_peak=2500.0,
        density=0.1,
        resolution=0.0025,
    ):
        self.width = float(width)
        self.height = float(height)
        self.safety_margin = float(safety_margin)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.smoothness = float(smoothness)
        self.sphere_factor = float(sphere_factor)
        self.normal_peak = float(normal_peak)
        self.shear_peak = float(shear_peak)
        self.density = float(density)
        self.resolution = float(resolution)

        # Same randomization as your teammate's force-map script.
        self.x0, self.y0 = np.random.uniform(0.03, 0.07, 2)
        self.rx, self.ry = np.random.uniform(0.02, 0.03, 2)

        self.side_range_x = np.arange(0.0, self.width + self.resolution, self.resolution)
        self.side_range_y = np.arange(0.0, self.height + self.resolution, self.resolution)

        self.x_grid, self.y_grid = np.meshgrid(self.side_range_x, self.side_range_y)

        self.normal_map, self.d_sq = generate_normal_force(
            self.x_grid,
            self.y_grid,
            self.x0,
            self.y0,
            self.rx,
            self.ry,
            smoothness=self.smoothness,
            sphere_factor=self.sphere_factor,
            peak_val=self.normal_peak,
        )

        shear_magnitude = generate_shear_force(
            self.x_grid,
            self.y_grid,
            self.d_sq,
            self.shear_peak,
        )

        dx = self.x0 - self.x_grid
        dy = self.y0 - self.y_grid

        distance = np.sqrt(dx ** 2 + dy ** 2)
        distance[distance == 0.0] = 1e-10

        self.shear_x_map = shear_magnitude * (dx / distance)
        shear_y_base = shear_magnitude * (dy / distance)

        r_avg = (self.rx + self.ry) / 2.0
        self.g_total = 150.0 * (r_avg / 0.03) ** 2 + np.random.uniform(-25.0, 25.0)

        self.shear_y_map = np.where(self.d_sq < 1.0, shear_y_base + (self.g_total * 200.0), 0.0,)

        # RegularGridInterpolator expects array coordinates as (y, x),
        # because the maps have shape [row=y, column=x].
        self.normal_interpolator = RegularGridInterpolator((self.side_range_y, self.side_range_x), self.normal_map, bounds_error=False, fill_value=0.0,)

        self.shear_x_interpolator = RegularGridInterpolator((self.side_range_y, self.side_range_x), self.shear_x_map, bounds_error=False, fill_value=0.0,)

        self.shear_y_interpolator = RegularGridInterpolator((self.side_range_y, self.side_range_x), self.shear_y_map, bounds_error=False, fill_value=0.0,)

    def __call__(self, x, y):
        
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

        points = np.column_stack(
            [
                y_array.ravel(),
                x_array.ravel(),
            ]
        )

        normal = self.normal_interpolator(points)
        shear_x = self.shear_x_interpolator(points)
        shear_y = self.shear_y_interpolator(points)

        normal = normal.reshape(x_array.shape)
        shear_x = shear_x.reshape(x_array.shape)
        shear_y = shear_y.reshape(x_array.shape)

        if normal.shape == ():
            return float(normal), float(shear_x), float(shear_y)

        return normal, shear_x, shear_y

    def get_sym_lim(self, data):
        limit = np.max(np.abs(data))

        if limit <= 0.0:
            limit = 1.0

        return -limit, limit

    def show(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        im1 = axes[0].pcolormesh(
            self.x_grid,
            self.y_grid,
            self.normal_map,
            cmap="viridis",
            shading="auto",
        )
        axes[0].set_title("Normale Kracht ($N/m^2$)")
        axes[0].set_aspect("equal")
        fig.colorbar(im1, ax=axes[0])

        lx, hx = self.get_sym_lim(self.shear_x_map)

        im2 = axes[1].pcolormesh(
            self.x_grid,
            self.y_grid,
            self.shear_x_map,
            cmap="RdBu_r",
            shading="auto",
            vmin=lx,
            vmax=hx,
        )
        axes[1].set_title("Schuifkracht X-Component\n(Wit = 0)")
        axes[1].set_aspect("equal")
        fig.colorbar(im2, ax=axes[1])

        ly, hy = self.get_sym_lim(self.shear_y_map)

        im3 = axes[2].pcolormesh(
            self.x_grid,
            self.y_grid,
            self.shear_y_map,
            cmap="RdBu_r",
            shading="auto",
            vmin=ly,
            vmax=hy,
        )
        axes[2].set_title(f"Schuifkracht Y-Component\n(Incl. {self.g_total:.1f}g)")
        axes[2].set_aspect("equal")
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()
