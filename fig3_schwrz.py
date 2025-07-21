import numpy as np
import matplotlib.pyplot as plt

def wrap_deg(angle_rad):
    """Wrap angle into [–90°, +90°]."""
    angle_wrapped = np.angle(np.exp(1j * angle_rad), deg=True)
    return ((angle_wrapped + 90) % 180) - 90

def split_segments(x, y, threshold=60):
    """Break into segments whenever |Δy| > threshold (for clean wrapping)."""
    segments = []
    start = 0
    for i in range(1, len(y)):
        if abs(y[i] - y[i - 1]) > threshold:
            segments.append((x[start:i], y[start:i]))
            start = i
    segments.append((x[start:], y[start:]))
    return segments

# Common φ grid
phi_deg = np.linspace(0, 360, 361)[:-1]
phi = np.deg2rad(phi_deg)

# Inclinations and radii to loop over
i_deg_list = [30, 60, 80]
r_list     = [3, 5, 15, 50]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, i_deg in zip(axes, i_deg_list):
    # inclination in radians
    i     = np.deg2rad(i_deg)
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    for r in r_list:
        u    = 1.0 / r
        beta = np.sqrt(u / (2 * (1 - u)))               # Eq. (6)

        # compute ψ
        cos_psi = np.clip(sin_i * np.cos(phi), -1, 1)
        psi     = np.arccos(cos_psi)
        sin_psi = np.sin(psi)

        # bending α (approx from Eq. 35)
        y     = 1 - cos_psi
        term1 = (u**2 * y**2) / 112
        term2 = (np.e * u * y / 100) * (np.log(1 - y/2) + y/2)
        cos_alpha = np.clip(
            1 - y * (1 - u) * (1 + term1 - term2),
            -1, 1
        )
        sin_alpha = np.sqrt(np.maximum(0, 1 - cos_alpha**2))

        # compute χ_GR (Eq. 29)
        tilde_a = (1 - cos_alpha * cos_psi) / (cos_alpha - cos_psi)
        chi_GR  = np.arctan2(cos_i * np.sin(phi),
                             tilde_a * sin_i + np.cos(phi))

        # compute χ_SR (Eq. 31)
        cos_zeta      = np.clip((sin_alpha / sin_psi) * cos_i, -1, 1)
        cos_xi        = np.clip(- (sin_alpha / sin_psi) * sin_i * np.sin(phi),
                                -1, 1)
        sin_zeta_sq   = 1 - cos_zeta**2
        numerator_SR   = -beta * cos_alpha * cos_zeta
        denominator_SR = sin_zeta_sq - beta * cos_xi
        chi_SR        = np.arctan2(numerator_SR, denominator_SR)

        # total rotation
        chi_tot_deg = wrap_deg(chi_GR + chi_SR)

        # plot with one legend entry per r
        first_segment = True
        line_color = None
        for seg_x, seg_y in split_segments(phi_deg, chi_tot_deg):
            if first_segment:
                ln, = ax.plot(seg_x, seg_y, label=f'$r={r}$')
                line_color = ln.get_color()
                first_segment = False
            else:
                ax.plot(seg_x, seg_y, color=line_color, label='_nolegend_')

    ax.set_title(f'Inclination $i={i_deg}^\\circ$')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.grid(linestyle=':', alpha=0.4)

# y‐label & legend on the first panel
axes[0].set_ylabel(r'$\chi^{\rm tot}$ (deg)')
axes[0].legend(loc='upper left', title='Schwarzschild $r$')

plt.tight_layout()
plt.savefig('fig3_schwarzschild.png', dpi=300)
plt.show()
