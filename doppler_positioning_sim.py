"""
Doppler Positioning Simulator — Transit-style pre-GPS satellite positioning.

Simulates the U.S. Navy NAVSAT (Transit) doppler-positioning method:
a polar-orbiting satellite broadcasts a stable carrier, the ground
receiver integrates Doppler shift over short counts, and a batch
least-squares solver inverts for receiver position. A first-order
Klobuchar-style ionospheric correction is applied.

Author: Dr. Mosab Hawarey (@DrHawarey)
License: MIT
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
MU_EARTH = 3.986004418e14     # m^3/s^2
R_EARTH = 6378137.0           # m, WGS84 semi-major axis
F_E = 1 / 298.257223563
C_LIGHT = 299792458.0         # m/s
F_CARRIER = 400e6             # Hz, NAVSAT-style 400 MHz carrier
OMEGA_E = 7.2921151467e-5     # rad/s, Earth rotation


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = F_E * (2 - F_E)
    N = R_EARTH / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h) * math.sin(lat)
    return np.array([x, y, z])


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = xyz
    e2 = F_E * (2 - F_E)
    lon = math.atan2(y, x)
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(8):
        N = R_EARTH / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e2 * N / (N + h)))
    N = R_EARTH / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h


def circular_orbit_state(t: np.ndarray, altitude: float, inclination_deg: float,
                          raan_deg: float, mean_anomaly0_deg: float) -> np.ndarray:
    """ECEF positions of a circular polar-orbiting satellite at times t (s)."""
    r = R_EARTH + altitude
    n = math.sqrt(MU_EARTH / r ** 3)
    inc = math.radians(inclination_deg)
    raan0 = math.radians(raan_deg)
    M0 = math.radians(mean_anomaly0_deg)

    pos = np.zeros((len(t), 3))
    for i, ti in enumerate(t):
        # ECI position in orbital plane
        u = M0 + n * ti
        x_op = r * math.cos(u)
        y_op = r * math.sin(u)
        # Rotate by inclination about x-axis, then by RAAN about z
        x_eci = math.cos(raan0) * x_op - math.sin(raan0) * math.cos(inc) * y_op
        y_eci = math.sin(raan0) * x_op + math.cos(raan0) * math.cos(inc) * y_op
        z_eci = math.sin(inc) * y_op
        # Rotate ECI -> ECEF by GMST = omega_e * t
        gmst = OMEGA_E * ti
        x_ecef = math.cos(gmst) * x_eci + math.sin(gmst) * y_eci
        y_ecef = -math.sin(gmst) * x_eci + math.cos(gmst) * y_eci
        z_ecef = z_eci
        pos[i] = (x_ecef, y_ecef, z_ecef)
    return pos


def doppler_count(receiver_ecef: np.ndarray, sat_ecef: np.ndarray, dt: float) -> np.ndarray:
    """Integrated Doppler count over interval dt — equals slant-range change / lambda."""
    rng = np.linalg.norm(sat_ecef - receiver_ecef, axis=1)
    drng = np.diff(rng)
    lam = C_LIGHT / F_CARRIER
    return drng / lam  # cycles per interval (negative if approaching)


def add_ionosphere(rng: np.ndarray, elevation: np.ndarray, vtec_tecu: float) -> np.ndarray:
    """Klobuchar-style slant ionospheric delay (m), single layer."""
    # Slant factor (Mannucci): 1 / cos(arcsin(R/(R+h) * cos(elev)))
    h_ion = 350e3
    factor = 1.0 / np.sqrt(1 - (R_EARTH / (R_EARTH + h_ion) * np.cos(elevation)) ** 2)
    # 40.3 / f^2 in m * TECU; 1 TECU = 1e16 el/m^2
    delay_vert = 40.3 / (F_CARRIER ** 2) * vtec_tecu * 1e16
    return delay_vert * factor


def elevation_from_ecef(rx: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Elevation angle of satellite from receiver (rad)."""
    rng_vec = sat - rx
    up = rx / np.linalg.norm(rx)
    cos_zen = np.sum(rng_vec * up, axis=1) / np.linalg.norm(rng_vec, axis=1)
    return np.arcsin(np.clip(cos_zen, -1, 1))


def solve_position_batch(sat_pos: np.ndarray, doppler_obs: np.ndarray, dt: float,
                          x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Batch least-squares for receiver ECEF position from Doppler counts.

    Observation model: count_i = (||sat_{i+1} - x|| - ||sat_i - x||) / lambda
    Linearized around x0, solved iteratively (Gauss-Newton).
    """
    lam = C_LIGHT / F_CARRIER
    x = x0.copy()
    for iteration in range(15):
        r1 = sat_pos[:-1] - x
        r2 = sat_pos[1:] - x
        d1 = np.linalg.norm(r1, axis=1)
        d2 = np.linalg.norm(r2, axis=1)
        model = (d2 - d1) / lam
        residual = doppler_obs - model
        # Jacobian: d(d2-d1)/dx = -r2/d2 + r1/d1
        J = (-r2 / d2[:, None] + r1 / d1[:, None]) / lam
        delta, *_ = np.linalg.lstsq(J, residual, rcond=None)
        x = x + delta
        if np.linalg.norm(delta) < 1e-3:
            break
    # Covariance
    Jt_J = J.T @ J
    try:
        cov = np.linalg.inv(Jt_J) * np.var(residual)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    return x, cov, iteration + 1


class DopplerSimApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Doppler Positioning Simulator — Transit-style")
        self.root.geometry("1180x780")
        self._build_ui()

    def _build_ui(self) -> None:
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Doppler Positioning Sim",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(left, text="True receiver position",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))

        self.lat_var = tk.DoubleVar(value=31.95)
        self.lon_var = tk.DoubleVar(value=35.91)
        self.h_var = tk.DoubleVar(value=780.0)
        for label, var in [("Latitude (deg)", self.lat_var),
                            ("Longitude (deg)", self.lon_var),
                            ("Height (m)", self.h_var)]:
            ttk.Label(left, text=label).pack(anchor="w")
            ttk.Entry(left, textvariable=var, width=15).pack(anchor="w")

        ttk.Label(left, text="Satellite pass",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))

        self.alt_var = tk.DoubleVar(value=1100e3)
        self.inc_var = tk.DoubleVar(value=90.0)
        self.raan_var = tk.DoubleVar(value=45.0)
        self.duration_var = tk.DoubleVar(value=900.0)
        self.dt_var = tk.DoubleVar(value=4.6)
        for label, var in [("Altitude (m)", self.alt_var),
                            ("Inclination (deg)", self.inc_var),
                            ("RAAN (deg)", self.raan_var),
                            ("Pass duration (s)", self.duration_var),
                            ("Count interval (s)", self.dt_var)]:
            ttk.Label(left, text=label).pack(anchor="w")
            ttk.Entry(left, textvariable=var, width=15).pack(anchor="w")

        ttk.Label(left, text="Errors",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))
        self.noise_var = tk.DoubleVar(value=0.05)
        self.tec_var = tk.DoubleVar(value=15.0)
        for label, var in [("Doppler noise (cycles)", self.noise_var),
                            ("Ionosphere TEC (TECU)", self.tec_var)]:
            ttk.Label(left, text=label).pack(anchor="w")
            ttk.Entry(left, textvariable=var, width=15).pack(anchor="w")

        ttk.Button(left, text="Simulate & Solve",
                   command=self._run).pack(pady=14, fill=tk.X)

        self.out = tk.Text(left, width=34, height=14, wrap="word",
                           font=("Consolas", 9))
        self.out.pack(fill=tk.BOTH)

        ttk.Label(left, text="Dr. Mosab Hawarey • MIT License",
                  font=("Segoe UI", 8), foreground="#777").pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run(self) -> None:
        try:
            rx_true = geodetic_to_ecef(self.lat_var.get(),
                                        self.lon_var.get(),
                                        self.h_var.get())
            dt = self.dt_var.get()
            t = np.arange(0.0, self.duration_var.get(), dt)
            sat = circular_orbit_state(t,
                                        altitude=self.alt_var.get(),
                                        inclination_deg=self.inc_var.get(),
                                        raan_deg=self.raan_var.get(),
                                        mean_anomaly0_deg=-30.0)
            elev = elevation_from_ecef(rx_true[None, :], sat)
            visible = elev > math.radians(7.5)
            if visible.sum() < 10:
                messagebox.showwarning("Sim", "Insufficient pass geometry — adjust orbit.")
                return

            sat = sat[visible]
            counts = doppler_count(rx_true, sat, dt)
            # Add ionospheric delay (affects range, hence count differences)
            elev_v = elevation_from_ecef(rx_true[None, :], sat)
            ion = add_ionosphere(np.linalg.norm(sat - rx_true, axis=1),
                                 elev_v, self.tec_var.get())
            lam = C_LIGHT / F_CARRIER
            counts = counts + np.diff(ion) / lam
            counts = counts + np.random.normal(0, self.noise_var.get(), size=counts.shape)

            x0 = rx_true + np.array([5_000.0, -3_000.0, 4_000.0])  # bad initial guess
            x_hat, cov, iters = solve_position_batch(sat, counts, dt, x0)
            lat_hat, lon_hat, h_hat = ecef_to_geodetic(x_hat)
            err = x_hat - rx_true
            err3d = float(np.linalg.norm(err))

            self.out.delete("1.0", tk.END)
            self.out.insert(tk.END,
                            f"Pass: {len(sat)} samples\n"
                            f"Solver: {iters} Gauss-Newton iters\n\n"
                            f"True:\n  lat {self.lat_var.get():.6f}\n"
                            f"  lon {self.lon_var.get():.6f}\n  h   {self.h_var.get():.1f} m\n\n"
                            f"Estimated:\n  lat {lat_hat:.6f}\n"
                            f"  lon {lon_hat:.6f}\n  h   {h_hat:.1f} m\n\n"
                            f"Error vector (m): ECEF\n"
                            f"  dX {err[0]:+.2f}\n  dY {err[1]:+.2f}\n  dZ {err[2]:+.2f}\n"
                            f"3-D error: {err3d:.2f} m\n")

            self.axes[0].clear()
            self.axes[0].plot(np.arange(len(counts)) * dt, counts, lw=0.9)
            self.axes[0].set_title("Doppler counts over pass")
            self.axes[0].set_xlabel("time (s)"); self.axes[0].set_ylabel("cycles per interval")
            self.axes[0].grid(alpha=0.3)

            self.axes[1].clear()
            self.axes[1].plot(np.arange(len(elev_v)) * dt, np.degrees(elev_v), color="C2")
            self.axes[1].set_title("Satellite elevation")
            self.axes[1].set_xlabel("time (s)"); self.axes[1].set_ylabel("elevation (deg)")
            self.axes[1].grid(alpha=0.3)
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Doppler Sim", f"Error:\n{exc}")
            raise


def main() -> None:
    root = tk.Tk()
    DopplerSimApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
