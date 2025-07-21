import numpy as np
import matplotlib.pyplot as plt

def compute_zeta_prime_wrapped(r, i_deg, phi_deg):
    """
    Compute the wrapped zenith angle ζ' (in degrees) for a given radius r and inclination i_deg,
    over an array of azimuths phi_deg.
    """
    u     = 1.0 / r
    i     = np.radians(i_deg)
    beta  = np.sqrt(u / (2 * (1 - u)))
    gamma = 1.0 / np.sqrt(1 - beta**2)
    phi   = np.radians(phi_deg)

    # Compute ψ
    cos_psi = np.sin(i) * np.cos(phi)
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    psi      = np.arccos(cos_psi)
    sin_psi  = np.sin(psi)

    # Bending angle α (approx from Eq. 35)
    y      = 1 - cos_psi
    term1  = (u**2 * y**2) / 112
    term2  = (np.e * u * y / 100) * (np.log(1 - y/2) + y/2)
    cos_alpha = 1 - y * (1 - u) * (1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha     = np.arccos(cos_alpha)
    sin_alpha = np.sqrt(np.maximum(0, 1 - cos_alpha**2))

    # Compute cos ξ and cos ζ
    cos_xi   = - (sin_alpha * np.sin(i) * np.sin(phi)) / sin_psi
    cos_zeta =   (sin_alpha * np.cos(i)) / sin_psi
    cos_xi   = np.clip(cos_xi,   -1.0, 1.0)
    cos_zeta = np.clip(cos_zeta, -1.0, 1.0)

    # Doppler factor δ and ζ'
    delta             = 1 / (gamma * (1 - beta * cos_xi))
    cos_zeta_prime    = delta * cos_zeta
    cos_zeta_prime    = np.clip(cos_zeta_prime, -1.0, 1.0)
    zeta_prime_deg    = np.degrees(np.arccos(cos_zeta_prime))

    # Wrap into [–90°, +90°]
    return ((zeta_prime_deg + 90) % 180) - 90

# Define azimuth grid, radii, and inclinations
phi_deg     = np.linspace(0, 360, 1000)
r_list      = [3, 5, 15, 50]
i_deg_list  = [30, 60, 80]

# Create 1×3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, i_deg in zip(axes, i_deg_list):
    for r in r_list:
        zeta_wrapped = compute_zeta_prime_wrapped(r, i_deg, phi_deg)
        ax.plot(phi_deg, zeta_wrapped, label=f'$r={r}$')

    ax.set_title(f'Inclination $i={i_deg}^\circ$')
    ax.set_xlabel(r'Azimuth $\varphi$ (deg)')
    ax.grid(linestyle=':', alpha=0.5)

# Common y‐label and legend on the first subplot
axes[0].set_ylabel(r"Zenith angle $\zeta'$ (deg)")
axes[0].legend(loc='upper left', title='Schwarzschild $r$')

# Uniform axis limits
plt.setp(axes, xlim=(0, 360))
plt.tight_layout()
plt.savefig('fig7_schwrz.png', dpi=300, bbox_inches='tight')
plt.show()
