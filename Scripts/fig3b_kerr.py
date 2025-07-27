import numpy as np
import matplotlib.pyplot as plt

def compute_beta(a, r):
    """
    Keplerian orbital speed β = v/c at radius r for spin parameter a.
    """
    B = 1 + a / np.sqrt(8 * r**3)
    D = 1 - 1/r + a**2 / (4 * r**2)
    F = 1 - a / np.sqrt(2 * r**3) + a**2 / (4 * r**3)
    return (F / (B * np.sqrt(D))) * np.sqrt(1 / (2 * r))

def zeta_prime_deg_at_r(i_deg, a, r):
    """
    Compute azimuth φ (deg) and zenith angle ζ' (deg, wrapped to [-90,90])
    for spin a, inclination i_deg, at a given radius r (in units of R_S).
    Returns (phi_deg_array, zeta_prime_deg_array).
    """
    # grid in azimuth
    phi_deg = np.linspace(0, 360, 1000)
    phi = np.deg2rad(phi_deg)

    # inclination
    i_rad = np.deg2rad(i_deg)
    sin_i, cos_i = np.sin(i_rad), np.cos(i_rad)

    # orbital β and Lorentz γ
    beta = compute_beta(a, r)
    gamma = 1.0 / np.sqrt(1 - beta**2)

    # angle ψ between disk normal and photon direction (Eq. 2)
    cos_psi = np.clip(sin_i * np.cos(phi), -1.0, 1.0)
    psi = np.arccos(cos_psi)
    sin_psi = np.sin(psi)

    # approximate light‐bending α (Eq. 35)
    u = 1.0 / r
    y = 1 - cos_psi
    term1 = (u**2 * y**2) / 112.0
    term2 = (np.e * u * y / 100.0) * (np.log(np.clip(1 - 0.5*y, 1e-10, None)) + 0.5*y)
    cos_alpha = 1 - y * (1 - u) * (1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    # special‐relativistic angles ξ and ζ (Eqs. 8,9)
    cos_xi   = np.clip(-(sin_alpha/sin_psi) * sin_i * np.sin(phi), -1.0, 1.0)
    cos_zeta = np.clip( (sin_alpha/sin_psi) * cos_i,               -1.0, 1.0)

    # Doppler factor δ and cos ζ' in fluid frame (Eq. 12)
    delta = 1.0 / (gamma * (1 - beta * cos_xi))
    cos_zeta_prime = delta * cos_zeta
    cos_zeta_prime = np.clip(cos_zeta_prime, -1.0, 1.0)

    # zenith angle in fluid frame and wrap to [-90,90]
    zeta_prime = np.degrees(np.arccos(cos_zeta_prime))
    zeta_prime_wrapped = ((zeta_prime + 90) % 180) - 90

    return phi_deg, zeta_prime_wrapped

# --- PLOTTING --- 
a = 0.8
r_values = [5, 3, 2]          # radii in R_S
inclinations = [30, 60, 80]   # degrees
colors = ['red', 'blue', 'green']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(rf"Zenith angle $\zeta'$ vs. $\varphi$, for $a={a}$", y=1.02)

for ax, r in zip(axes, r_values):
    for color, i_deg in zip(colors, inclinations):
        φ, ζp = zeta_prime_deg_at_r(i_deg, a, r)
        ax.plot(φ, ζp, color=color, label=f"$i={i_deg}^\\circ$")
    ax.set_title(rf"$r = {r}\,R_S$")
    ax.set_xlabel(r"$\varphi\ (\mathrm{{deg}})$")
    ax.grid(linestyle=':', alpha=0.5)
    ax.legend(title="Inclination")

axes[0].set_ylabel(r"$\zeta'\ (\mathrm{deg})$")
plt.tight_layout()
plt.savefig("fig3b_kerr.png", dpi=300)
plt.show()
