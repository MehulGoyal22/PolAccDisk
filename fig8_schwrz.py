import numpy as np
import matplotlib.pyplot as plt

def compute_zeta_prime_deg(r, i_deg=80):
    u = 1.0 / r
    i = np.radians(i_deg)
    beta = np.sqrt(u / (2 * (1 - u)))
    gamma = 1.0 / np.sqrt(1 - beta**2)

    phi_deg = np.linspace(0, 360, 1000)
    phi = np.radians(phi_deg)

    cos_psi = np.sin(i) * np.cos(phi)
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    psi = np.arccos(cos_psi)
    sin_psi = np.sin(psi)
 
    # Compute alpha using Eq. 35
    y = 1 - cos_psi
    term1 = (u**2 * y**2) / 112
    term2 = (np.e * u * y / 100) * (np.log(1 - y / 2) + y / 2)
    cos_alpha = 1 - y * (1 - u) * (1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    sin_alpha = np.sqrt(np.maximum(0, 1 - cos_alpha**2))

    # Compute cos(xi) and cos(zeta)
    cos_xi = - (sin_alpha * np.sin(i) * np.sin(phi)) / sin_psi
    cos_zeta = (sin_alpha * np.cos(i)) / sin_psi
    cos_xi = np.clip(cos_xi, -1.0, 1.0)
    cos_zeta = np.clip(cos_zeta, -1.0, 1.0)

    # Doppler factor and cos(zeta')
    delta = 1 / (gamma * (1 - beta * cos_xi))
    g = delta * (np.sqrt(1 - u))
    cos_zeta_prime = delta * cos_zeta
    cos_zeta_prime = np.clip(cos_zeta_prime, -1.0, 1.0)
    zeta_prime_deg = np.degrees(np.arccos(cos_zeta_prime))

    # Light bending
    term_1 = 3*(u**2 * y**2) / 112
    term_2 = (np.e * u * y / 100)
    term_3 = 2*np.log(1 - y / 2)
    term_4 = y*((1-3*y/4)/(1-y/2))
    L = 1 + term_1 - term_2*(term_3 + term_4)

    flux_values = (g**3) * L * cos_zeta

    return phi_deg, ((zeta_prime_deg + 90) % 180) - 90, flux_values

# Radii and plotting settings
radii  = [3, 5, 15, 50]
colors = ['black', 'red', 'blue', 'green']
labels = [f"r = {r}" for r in radii]
i_list = [30, 60, 80]

# Create 1×3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, i_deg in zip(axes, i_list):
    for r, c, lbl in zip(radii, colors, labels):
        phi_vals, _, flux_vals = compute_zeta_prime_deg(r, i_deg)
        ax.plot(phi_vals, flux_vals, color=c, label=lbl)
    ax.set_title(f"Inclination $i={i_deg}^\\circ$")
    ax.set_xlabel(r"Azimuthal angle $\varphi$ (deg)")
    ax.grid(True, linestyle=':')

# Common y-label and legend on first subplot
axes[0].set_ylabel(r"$g^{3} L \, \cos\zeta$")
axes[0].legend(loc='upper right', title="Schwarzschild $r$")
plt.setp(axes, xlim=(0, 360))

plt.tight_layout()
plt.savefig('fig8_schwrz.png', dpi=300, bbox_inches='tight')
plt.show()
