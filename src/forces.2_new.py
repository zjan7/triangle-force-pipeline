import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

def _generate_normal_force(x, y, x0, y0, rx, ry, smoothness=1.0, sphere_factor=1.2, peak_val=1.0):
    dx = x - x0
    dy = y - y0
    angle = np.arctan2(dy, dx)

    # Perturb radius with a few random harmonics
    n_harmonics = 4
    harmonics = np.zeros_like(angle)
    for k in range(1, n_harmonics + 1):
        amp = np.random.uniform(0.0, 0.15)
        phase = np.random.uniform(0.0, 2 * np.pi)
        harmonics += amp * np.sin(k * angle + phase)

    rx_eff = rx * (1.0 + harmonics)
    ry_eff = ry * (1.0 + harmonics)

    d_sq = (dx ** 2 / rx_eff ** 2) + (dy ** 2 / ry_eff ** 2)
    d_sq = np.clip(d_sq, 0.0, 1.0)

    z_norm = np.sqrt(1.0 - d_sq ** sphere_factor)
    z_norm[d_sq >= 1.0] = 0.0

    noise = np.random.normal(0.0, 0.02, x.shape)
    z_norm = np.where(z_norm > 0.0, z_norm + noise, 0.0)

    z_norm_smooth = gaussian_filter(z_norm, sigma=smoothness)
    z_norm_smooth[d_sq > 0.95] = 0.0
    z_norm_smooth = np.maximum(z_norm_smooth, 0.0)

    return z_norm_smooth * peak_val, d_sq


def _generate_shear_force(x, y, d_sq, peak_val):
    raw_profile = (d_sq ** 0.3) * ((1.0 - d_sq) ** 0.3)

    max_raw = np.max(raw_profile)

    if max_raw <= 0.0:
        max_raw = 1.0

    normalized_profile = raw_profile / max_raw

    shear_noise = np.random.normal(0.0, 0.01, x.shape)

    shear_val = np.where(
        d_sq < 1.0,
        (normalized_profile + shear_noise) * peak_val,
        0.0,
    )

    return np.maximum(shear_val, 0.0)


class ForceGenerator2:
  
    def __init__(
        self,
        width=0.1,
        height=0.1,
        safety_margin=0.03,
        r_min=0.02,
        r_max=0.05,
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
        self.x0 = np.random.uniform(r_max, width - r_max, 1)
        self.y0 = np.random.uniform(r_max, height - r_max, 1)
        self.rx, self.ry = np.random.uniform(r_min, r_max, 2)

        self.magic1 = np.random.choice([-1, 1])
        self.magic2 = np.random.choice([-1, 1])
        self.magic3 = np.random.uniform(0.2, 1)

        self.side_range_x = np.arange(0.0, self.width + self.resolution, self.resolution)
        self.side_range_y = np.arange(0.0, self.height + self.resolution, self.resolution)

        self.x_grid, self.y_grid = np.meshgrid(self.side_range_x, self.side_range_y)

        self.normal_map, self.d_sq = _generate_normal_force(
            self.x_grid,
            self.y_grid,
            self.x0,
            self.y0,
            self.rx,
            self.ry,
            smoothness=self.smoothness,
            sphere_factor=self.sphere_factor,
            peak_val=self.normal_peak * self.magic3,
        )

        shear_magnitude = _generate_shear_force(
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
        self.g_total = 0 # zwaartekracht uit. Verpest momenteel simulaties

        self.shear_y_map = np.where(
            self.d_sq < 1.0,
            shear_y_base + (self.g_total * 200.0),
            0.0,
        )

        # RegularGridInterpolator expects array coordinates as (y, x),
        # because the maps have shape [row=y, column=x].
        self.normal_interpolator = RegularGridInterpolator(
            (self.side_range_y, self.side_range_x),
            self.normal_map,
            bounds_error=False,
            fill_value=0.0,
        )

        self.shear_x_interpolator = RegularGridInterpolator(
            (self.side_range_y, self.side_range_x),
            self.shear_x_map,
            bounds_error=False,
            fill_value=0.0,
        )

        self.shear_y_interpolator = RegularGridInterpolator(
            (self.side_range_y, self.side_range_x),
            self.shear_y_map,
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, x, y):
        """
        Return force densities at one point or many points.

        For scalar x, y:
            returns three floats

        For array-like x, y:
            returns three arrays
        """
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

        return normal, self.magic2 * shear_x, shear_y * self.magic1

    def get_sym_lim(self, data):
        limit = np.max(np.abs(data))

        if limit <= 0.0:
            limit = 1.0

        return -limit, limit

    def show(self):
        normal, shear_x, shear_y = self(self.x_grid, self.y_grid)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        im1 = axes[0].pcolormesh(
            self.x_grid,
            self.y_grid,
            normal,
            cmap="viridis",
            shading="auto",
        )
        axes[0].set_title("Normale Kracht ($N/m^2$)")
        axes[0].set_aspect("equal")
        fig.colorbar(im1, ax=axes[0])

        lx, hx = self.get_sym_lim(shear_x)

        im2 = axes[1].pcolormesh(
            self.x_grid,
            self.y_grid,
            shear_x,
            cmap="RdBu_r",
            shading="auto",
            vmin=lx,
            vmax=hx,
        )
        axes[1].set_title("Schuifkracht X-Component\n(Wit = 0)")
        axes[1].set_aspect("equal")
        fig.colorbar(im2, ax=axes[1])

        ly, hy = self.get_sym_lim(shear_y)

        im3 = axes[2].pcolormesh(
            self.x_grid,
            self.y_grid,
            shear_y,
            cmap="RdBu_r",
            shading="auto",
            vmin=ly,
            vmax=hy,
        )
        axes[2].set_title(
            f"Schuifkracht Y-Component\n(Incl. {self.g_total:.1f}g)"
        )
        axes[2].set_aspect("equal")
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()
