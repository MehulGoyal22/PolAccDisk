import numpy as np
import matplotlib.pyplot as plt

# Common parameters
r = 3.0                   # radius in Schwarzschild units
u = 1.0 / r
phi_deg = np.linspace(0, 360, 361)[:-1]  # exclude duplicate endpoint
phi = np.deg2rad(phi_deg)

def wrap_deg(angle_rad):
    angle_wrapped = np.angle(np.exp(1j * angle_rad), deg=True)
    return ((angle_wrapped + 90) % 180) - 90

def split_segments(x, y, threshold=60):
    segments = []
    start = 0
    for i in range(1, len(y)):
        if abs(y[i] - y[i - 1]) > threshold:
            segments.append((x[start:i], y[start:i]))
            start = i
    segments.append((x[start:], y[start:]))
    return segments

# Inclinations to plot
i_deg_list = [30, 60, 80]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, i_deg in zip(axes, i_deg_list):
    # Convert inclination
    i = np.deg2rad(i_deg)
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    # Orbital velocity (beta) from Eq. (6)
    beta = np.sqrt(u / (2 * (1 - u)))

    # Compute psi
    cos_psi = sin_i * np.cos(phi)
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    psi = np.arccos(cos_psi)
    sin_psi = np.sin(psi)

    # Bending angle alpha (Eq. 35 approximation)
    y = 1 - cos_psi
    term1 = (u**2 * y**2) / 112
    term2 = (np.e * u * y / 100) * (np.log(1 - y / 2) + y / 2)
    cos_alpha = 1 - y * (1 - u) * (1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    # Angles for SR terms
    cos_zeta = (np.sin(alpha) / sin_psi) * cos_i
    cos_xi   = - (np.sin(alpha) / sin_psi) * sin_i * np.sin(phi)
    cos_zeta = np.clip(cos_zeta, -1.0, 1.0)
    cos_xi   = np.clip(cos_xi,   -1.0, 1.0)
    sin_zeta = np.sqrt(np.maximum(0, 1 - cos_zeta**2))

    # chi_GR (Eq. 29)
    tilde_a = (1 - cos_alpha * cos_psi) / (cos_alpha - cos_psi)
    chi_GR = np.arctan2(cos_i * np.sin(phi), tilde_a * sin_i + np.cos(phi))

    # chi_SR (Eq. 31)
    numerator_SR   = -beta * cos_alpha * cos_zeta
    denominator_SR = sin_zeta**2 - beta * cos_xi
    chi_SR = np.arctan2(numerator_SR, denominator_SR)

    # chi_SR_flat (Eq. 32)
    numerator_flat   = -beta * cos_i * np.cos(phi)
    denominator_flat = sin_i + beta * np.sin(phi)
    chi_SR_flat = np.arctan2(numerator_flat, denominator_flat)

    # Total rotation
    chi_tot = chi_GR + chi_SR

    # Wrap to [–90°, +90°]
    chi_GR_deg      = wrap_deg(chi_GR)
    chi_SR_deg      = wrap_deg(chi_SR)
    chi_SR_flat_deg = wrap_deg(chi_SR_flat)
    chi_tot_deg     = wrap_deg(chi_tot)

    # Plot each in smooth segments
    for phi_x, chi_y in split_segments(phi_deg, chi_GR_deg):
        ax.plot(phi_x, chi_y, 'r-.', label=r'$\chi^{\rm GR}$' if phi_x[0]==phi_deg[0] else "")
    for phi_x, chi_y in split_segments(phi_deg, chi_SR_deg):
        ax.plot(phi_x, chi_y, 'b--', label=r'$\chi^{\rm SR}$' if phi_x[0]==phi_deg[0] else "")
    for phi_x, chi_y in split_segments(phi_deg, chi_SR_flat_deg):
        ax.plot(phi_x, chi_y, 'g:', label=r'$\chi^{\rm SR}_{\rm flat}$' if phi_x[0]==phi_deg[0] else "")
    for phi_x, chi_y in split_segments(phi_deg, chi_tot_deg):
        ax.plot(phi_x, chi_y, 'k-', label=r'$\chi^{\rm tot}$' if phi_x[0]==phi_deg[0] else "")

    ax.set_title(f'inclination $i={i_deg}°$')
    ax.grid(linestyle=':', alpha=0.5)

# Common labels and legend on the first subplot
axes[0].set_xlabel(r'$\varphi$ (deg)')
axes[0].set_ylabel(r'Polarization rotation $\chi$ (deg)')
axes[1].set_xlabel(r'$\varphi$ (deg)')
axes[2].set_xlabel(r'$\varphi$ (deg)')
axes[0].legend(loc='upper left')

# Set shared limits
plt.setp(axes, xlim=(0,360), ylim=(-90,90))

plt.tight_layout()
plt.savefig('fig2_schwarzschild.png', dpi=300)

plt.show()
