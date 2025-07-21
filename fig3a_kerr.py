import numpy as np
import matplotlib.pyplot as plt

# Constants
spins = [0.2, 0.5, 0.8]
inclinations = [30, 60, 80]  # in degrees
colors = {30: 'red', 60: 'green', 80: 'blue'}
phi_deg = np.linspace(0, 360, 361)
phi_rad = np.deg2rad(phi_deg)

# ISCO radius from spin
def compute_r_isco(a):
    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 0.5 * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))  # prograde orbit

# Orbital beta
def compute_beta(a, r):
    B = 1 + a / np.sqrt(8 * r**3)
    D = 1 - 1/r + a**2 / (4 * r**2)
    F = 1 - a / np.sqrt(2 * r**3) + a**2 / (4 * r**3)
    return (F / (B * np.sqrt(D))) * np.sqrt(1 / (2 * r))

# Poutanen 2020 light bending: cos(alpha)
def cos_alpha_poutanen(cos_psi, r):
    u = 1 / r
    y = 1 - cos_psi
    e = np.e
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (u**2 * y**2) / 112.
        term2 = (e / 100.) * u * y * (np.log(1 - y / 2) + y / 2)
        return 1 - (1 - u) * y * (1 + term1 - term2)

# Start plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True)

for col, a in enumerate(spins):
    r = compute_r_isco(a)
    beta = compute_beta(a, r)

    for i in inclinations:
        i_rad = np.deg2rad(i)
        cos_psi = np.sin(i_rad) * np.cos(phi_rad)
        cos_alpha_vals = cos_alpha_poutanen(cos_psi, r)
        cos_alpha_vals = np.clip(cos_alpha_vals, -1, 1)
        sin_alpha_vals = np.sqrt(1 - cos_alpha_vals**2)
        sin_psi = np.sqrt(1 - cos_psi**2)

        # Emission angle in lab frame
        cos_zeta = (sin_alpha_vals / sin_psi) * np.cos(i_rad)
        cos_zeta = np.clip(cos_zeta, -1, 1)

        # Doppler factor δ and cos(ξ)
        gamma = 1 / np.sqrt(1 - beta**2)
        cos_xi = -(sin_alpha_vals / sin_psi) * np.sin(i_rad) * np.sin(phi_rad)
        delta = 1 / (gamma * (1 - beta * cos_xi))

        # Emission angle in comoving frame
        cos_zeta_prime = delta * cos_zeta
        cos_zeta_prime = np.clip(cos_zeta_prime, -1, 1)
        zeta_prime = np.degrees(np.arccos(cos_zeta_prime))

        # Plot ζ'
        axes[col].plot(phi_deg, zeta_prime, color=colors[i], label=f'{i}°', lw=1.5)

    # Annotate panel
    axes[col].text(0.05, 0.9, f'a = {a}, r = {r:.2f}', transform=axes[col].transAxes)
    axes[col].text(0.95, 0.9, f'({chr(97+col)})', transform=axes[col].transAxes,
                   ha='right', va='top')

# Configure axes
for ax in axes:
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 90)
    ax.set_ylabel(r'$\zeta^\prime$ (deg)')
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.show()
