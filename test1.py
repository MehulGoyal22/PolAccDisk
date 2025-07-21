import numpy as np
import matplotlib.pyplot as plt

# Parameters
inclinations = [30, 60, 80]          # Inclination angles in degrees
r_values = np.linspace(3, 15, 1000)   # Radii
phi_values = np.linspace(0, 2*np.pi, 1200, endpoint=False)  # Azimuth
R_S = 1.0                            # Schwarzschild radius

# Polarization stick locations
r_sticks = np.array([3, 6, 9, 12])
phi_sticks = np.deg2rad([0, 60, 120, 180, 240, 300])

# Atmosphere functions
def a_es(cos_zeta): return 0.421 + 0.868 * cos_zeta
def p_es(cos_zeta): return 0.1171 * (1 - cos_zeta) / (1 + 3.5 * cos_zeta)
def t_power4(u): return np.where(u <= 1/3, u**3 * (1 - np.sqrt(3 * u)), 0)

def g_factor(u, sin_i, sin_phi,sin_alpha,sin_psi):
    beta = np.sqrt(0.5 * (u/(1-u))) 
    gamma = 1.0 / np.sqrt(1 - beta**2)
    denom = (1 + beta * sin_i * sin_phi * sin_alpha / sin_psi)
    return np.sqrt(1 - 1.5*u) / (denom)

# Light-bending projection
def project_to_sky(r, phi, i_rad):
    u = R_S / r
    beta = np.sqrt(0.5 * (u/(1-u)))
    gamma = 1.0 / np.sqrt(1 - beta**2)
    sin_i, cos_i = np.sin(i_rad), np.cos(i_rad)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_psi = sin_i * cos_phi
    sin_psi = np.sqrt(np.maximum(0.0, 1 - cos_psi**2))
    y = 1 - cos_psi

    # Light bending
    e_const = np.e
    cos_alpha_approx = 1 - y * (1 - u) * (1 + (u**2 * y**2)/112 -
        (e_const * u * y / 100) * (np.log(1 - 0.5 * y) + 0.5 * y))
    cos_alpha_approx = np.clip(cos_alpha_approx, -1, 1)
    sin_alpha_approx = np.sqrt(np.clip(1 - cos_alpha_approx**2, 0, 1))
    b_approx = (r / np.sqrt(1 - u)) * sin_alpha_approx

    # cos_alpha_exact = u + (1 - u) * cos_psi
    # cos_alpha_exact = np.clip(cos_alpha_exact, -1, 1)
    # sin_alpha_exact = np.sqrt(np.clip(1 - cos_alpha_exact**2, 0, 1))
    # b_exact = r / np.sqrt(1 - u) * sin_alpha_exact

    # Sky plane coordinates: X = left-right (azimuth), Y = up-down (vertical)
    sin_psi_safe = np.where(sin_psi == 0, 1e-6, sin_psi)
    cosPhi = -cos_i * cos_phi / sin_psi_safe
    sinPhi = -sin_phi / sin_psi_safe

    # FLIP: X = b * cosPhi (horizontal); Y = b * sinPhi (vertical)
    # X_exact = b_exact * cosPhi
    # Y_exact = b_exact * sinPhi
    X_approx = b_approx * cosPhi
    Y_approx = -b_approx * sinPhi
    

    cos_zeta = (sin_alpha_approx / sin_psi) * cos_i
    denom = (1 + beta * sin_i * sin_phi * sin_alpha_approx / sin_psi)
    delta = 1/(gamma * denom)
    cos_zeta_prime = delta * cos_zeta


    return X_approx, Y_approx, sin_alpha_approx, sin_psi_safe, cos_zeta_prime, cos_zeta


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, inc in zip(axes, inclinations):
    i_rad = np.deg2rad(inc)

    rr, pp = np.meshgrid(r_values, phi_values, indexing='xy')
    X_a, Y_a,sin_alpha_val,sin_psi_val,cos_zeta_prime,cos_zeta = project_to_sky(rr, pp, i_rad)
    g = g_factor(R_S/rr, np.sin(i_rad), np.sin(pp), sin_alpha_val, sin_psi_val)
    I = (g**4) * t_power4(R_S/rr) * a_es(cos_zeta_prime)*cos_zeta

    # Histogram for image: swap X and Y to make disk horizontal
    Npix = 1000
    grid = np.linspace(-16, 16, Npix+1)
    H, xedges, yedges = np.histogram2d(Y_a.ravel(), X_a.ravel(),
                                       bins=[grid, grid], weights=I.ravel())
    H[H <= 0] = 1e-12  # Avoid log(0)
    logH = np.log10(H / H.max())
    vmin = np.percentile(logH[H > 1e-12], 1)  # 1st percentile of non-zero values
    vmax = 0

    ax.imshow(logH.T, origin='lower', extent=[-16, 16, -16, 16],
          cmap='inferno', aspect='equal', vmin=vmin, vmax=vmax)



    # Contours of constant radius
    for r_val in [3, 6, 9, 12, 15]:
        phic = np.linspace(0, 2*np.pi, 400)
        X_a_r, Y_a_r, sin_alpha_r, sin_psi_r, cos_zeta_r, cos_zeta_r_og = project_to_sky(np.full_like(phic, r_val), phic, i_rad)
        ax.plot(Y_a_r, X_a_r, 'w--', lw=0.5)

    # Contours of constant azimuth
    for phi_deg in range(0, 360, 30):
        phi_line = np.deg2rad(phi_deg)
        rr_line = np.linspace(3, 15, 200)
        X_a_p, Y_a_p, sin_alpha_p, sin_psi_p, cos_zeta_p, cos_zeta_p_og = project_to_sky(rr_line, np.full_like(rr_line, phi_line), i_rad)
        ax.plot(Y_a_p, X_a_p, 'w--', lw=0.5)

    # Polarization vectors
    X_a_s, Y_a_s, sin_alpha_s, sin_psi_s, cos_zeta_s, cos_zeta_s_og = project_to_sky(r_sticks[:, None], phi_sticks[None, :], i_rad)
    PD = p_es(cos_zeta_s)
    length = 8.0 * PD

    X_a_flat, Y_a_flat = X_a_s.flatten(), Y_a_s.flatten()
    length_flat = length.flatten()
    phi_mesh = np.tile(phi_sticks, len(r_sticks))
    angle = phi_mesh + np.pi/2
    dx = length_flat * np.cos(angle)
    dy = length_flat * np.sin(angle)


    for (xa, ya, dx_i, dy_i) in zip(X_a_flat, Y_a_flat, dx, dy):
        ax.plot([ya, ya + dy_i], [xa, xa + dx_i], color='blue', lw=1)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_title(f"$i = {inc}^\\circ$")
    ax.axis('off')  # optional

plt.tight_layout()
plt.show()