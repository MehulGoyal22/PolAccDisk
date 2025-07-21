import numpy as np
import matplotlib.pyplot as plt

def compute_r_isco(a):
    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 0.5 * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))  # prograde

def compute_beta(a, r):
    B = 1 + a / np.sqrt(8 * r**3)
    D = 1 - 1/r + a**2 / (4 * r**2)
    F = 1 - a / np.sqrt(2 * r**3) + a**2 / (4 * r**3)
    return (F / (B * np.sqrt(D))) * np.sqrt(1 / (2 * r))

def chi_total_func(a, i_deg):
    """
    Returns: phi_deg (array), chi_tot (radians array), r_isco (scalar)
    at r = r_isco for spin a.
    """
    # r = r_isco
    r_isco = compute_r_isco(a)
    r = r_isco

    # inclination
    i = np.deg2rad(i_deg)
    sin_i, cos_i = np.sin(i), np.cos(i)

    # orbital speed
    beta = compute_beta(a, r)

    # azimuth grid
    phi_deg = np.linspace(0, 360, 361)[:-1]
    phi = np.deg2rad(phi_deg)

    # compute ψ
    cos_psi = np.clip(sin_i * np.cos(phi), -1, 1)
    psi = np.arccos(cos_psi)
    sin_psi = np.sin(psi)

    # bending α (approx Eq.35)
    u = 1.0 / r
    y = 1 - cos_psi
    term1 = (u**2 * y**2) / 112
    term2 = (np.e * u * y / 100) * (np.log(1 - 0.5*y) + 0.5*y)
    cos_alpha = 1 - y*(1-u)*(1 + term1 - term2)
    cos_alpha = np.clip(cos_alpha, -1, 1)
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    # angles for SR
    cos_zeta = np.clip((sin_alpha/sin_psi)*cos_i, -1, 1)
    cos_xi   = np.clip(- (sin_alpha/sin_psi)*sin_i*np.sin(phi), -1, 1)
    sin_zeta = np.sqrt(1 - cos_zeta**2)

    # χ_GR (Eq.29–30)
    tilde_a = (1 - cos_alpha*cos_psi) / (cos_alpha - cos_psi)
    chi_GR  = np.arctan2(cos_i*np.sin(phi),
                        tilde_a*sin_i + np.cos(phi))

    # χ_SR (Eq.31)
    num_SR = -beta * cos_alpha * cos_zeta
    den_SR = sin_zeta**2 - beta * cos_xi
    chi_SR  = np.arctan2(num_SR, den_SR)

    chi_tot = chi_GR + chi_SR
    return phi_deg, chi_tot, r_isco

def wrap_deg(angle_rad):
    deg = np.rad2deg(angle_rad)
    w = np.angle(np.exp(1j*np.deg2rad(deg)), deg=True)
    return ((w + 90) % 180) - 90

def split_segments(x, y, threshold=80):
    segs, start = [], 0
    for i in range(1, len(y)):
        if abs(y[i] - y[i-1]) > threshold:
            segs.append((x[start:i], y[start:i]))
            start = i
    segs.append((x[start:], y[start:]))
    return segs

# --- PLOTTING ---
a_values     = [0.2, 0.5, 0.8]
inclinations = [30, 60, 80]
colors       = ['red', 'green', 'blue']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, a_val in zip(axes, a_values):
    # Compute r_isco once for annotation
    _, _, r_isco = chi_total_func(a_val, inclinations[0])

    for i_deg, col in zip(inclinations, colors):
        phi_vals, chi_vals, _ = chi_total_func(a_val, i_deg)
        chi_deg = wrap_deg(chi_vals)

        first = True
        for seg_x, seg_y in split_segments(phi_vals, chi_deg):
            if first:
                ax.plot(seg_x, seg_y, color=col,
                        label=f"$i={i_deg}^\\circ$")
                first = False
            else:
                ax.plot(seg_x, seg_y, color=col)

    # Annotate spin and ISCO radius
    ax.text(0.98, 0.02,
            rf"$a={a_val},\ r_{{\rm isco}}={r_isco:.3f}\,R_S$",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="gray", alpha=0.8))

    ax.set_xlim(0, 360)
    ax.set_xlabel(r"$\varphi\ (\rm deg)$")
    if ax is axes[0]:
        ax.set_ylabel(r"$\chi_{\rm tot}\ (\rm deg)$")
    ax.set_title(rf"$a={a_val}$")
    ax.grid(linestyle=':', color='gray', alpha=0.5)
    ax.legend(title="Inclination")

plt.tight_layout()
plt.savefig("fig2a_kerr.png", dpi=300)
plt.show()
