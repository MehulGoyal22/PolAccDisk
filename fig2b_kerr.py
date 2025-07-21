import numpy as np
import matplotlib.pyplot as plt

def compute_r_isco(a):
    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 0.5 * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))

def compute_beta(a, r):
    B = 1 + a / np.sqrt(8 * r**3)
    D = 1 - 1/r + a**2 / (4 * r**2)
    F = 1 - a / np.sqrt(2 * r**3) + a**2 / (4 * r**3)
    return (F / (B * np.sqrt(D))) * np.sqrt(1 / (2 * r))

def zeta_prime_deg(i_deg, a):
    """
    Compute φ (deg) and ζ' (deg, wrapped to [-90,90]) for spin a,
    inclination i_deg, at r = r_isco(a).
    """
    r = compute_r_isco(a)
    i = np.radians(i_deg)
    sin_i, cos_i = np.sin(i), np.cos(i)
    beta = compute_beta(a, r)
    gamma = 1/np.sqrt(1 - beta**2)

    phi_deg = np.linspace(0, 360, 1000)
    phi = np.radians(phi_deg)

    cos_psi = np.clip(sin_i * np.cos(phi), -1, 1)
    psi = np.arccos(cos_psi)
    sin_psi = np.sin(psi)

    u = 1.0 / r
    y = 1 - cos_psi
    term1 = (u**2 * y**2)/112
    term2 = (np.e * u * y/100)*(np.log(1 - 0.5*y) + 0.5*y)
    cos_alpha = 1 - y*(1-u)*(1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1, 1)
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    cos_xi = - (sin_alpha * sin_i * np.sin(phi)) / sin_psi
    cos_zeta =   (sin_alpha * cos_i)       / sin_psi
    cos_xi   = np.clip(cos_xi,   -1, 1)
    cos_zeta = np.clip(cos_zeta, -1, 1)

    delta = 1/(gamma*(1 - beta*cos_xi))
    cos_zeta_p = delta * cos_zeta
    cos_zeta_p = np.clip(cos_zeta_p, -1, 1)
    zeta_p = np.degrees(np.arccos(cos_zeta_p))
    # wrap to [-90,90]
    zeta_p = ((zeta_p + 90) % 180) - 90

    return phi_deg, zeta_p, r

# plotting
a_values = [0.2, 0.5, 0.8]
inclinations = [30, 60, 80]
colors = ['red', 'blue', 'green']

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, a in zip(axes, a_values):
    # we'll use r_isco for the annotation
    _, _, r_isco = zeta_prime_deg(inclinations[0], a)

    for i_deg, color in zip(inclinations, colors):
        φ, ζp, _ = zeta_prime_deg(i_deg, a)
        ax.plot(φ, ζp, color=color, label=f"$i={i_deg}^\\circ$")

    ax.set_title(rf"$a={a},\ r_{{\rm isco}}={r_isco:.3f}\,R_S$")
    ax.set_xlabel(r"$\varphi\ (\mathrm{deg})$")
    ax.grid(linestyle=':', alpha=0.5)
    ax.legend(title="Inclination")

axes[0].set_ylabel(r"$\zeta'\ (\mathrm{deg})$")
plt.tight_layout()
plt.savefig("fig2b_kerr.png", dpi=300)
plt.show()
