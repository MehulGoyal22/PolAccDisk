import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Per-panel quiver scale factors
scale_factor_dic = {30: 80, 60: 40, 80: 20}

def plot_accretion_disk_image(ax, i_deg):
    i = np.deg2rad(i_deg)

    # Build (r, phi) grid
    r_min, r_max, n_r = 1, 15, 400
    phi_min, phi_max, n_phi = 0, 2*np.pi, 400
    r = np.linspace(r_min, r_max, n_r)
    phi = np.linspace(phi_min, phi_max, n_phi)
    r_grid, phi_grid = np.meshgrid(r, phi)
    u = 1.0 / r_grid

    # Physics computations
    cos_psi   = np.sin(i) * np.cos(phi_grid)
    sin_psi   = np.sqrt(1 - cos_psi**2)
    y         = 1 - cos_psi
    log_term  = np.log1p(-y/2)
    cos_alpha = 1 - y*(1-u)*(1 + (u**2*y**2)/112 
                          - (np.e*u*y/100)*(log_term + y/2))
    sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))

    beta  = np.sqrt(u/(2*(1-u)))
    gamma = np.sqrt((1-u)/(1 - 1.5*u))

    cos_zeta       = (sin_alpha/sin_psi)*np.cos(i)
    sin_zeta       = np.sqrt(1 - cos_zeta**2)
    cos_xi         = -(sin_alpha/sin_psi)*np.sin(i)*np.sin(phi_grid)
    delta          = 1/(gamma*(1 - beta*cos_xi))
    g              = delta*np.sqrt(1 - u)
    cos_zeta_prime = delta*cos_zeta

    t_u_fourth    = u**3 * (np.sqrt(1 - u))
    numerator = 1 + 5.3*cos_zeta_prime - 0.2*cos_zeta_prime**2
    denominator = 1 + 1.3*cos_zeta_prime + 4.4*cos_zeta_prime**2
    a_es          = (1.73*cos_zeta_prime)*(numerator/denominator)

    # Intensity map for different radii
    r_grid = np.clip(r_grid, 1, 15)  # Avoid division by zero
    I_map = np.zeros_like(r_grid)
    I_map[r_grid <= 5] = (g**4)*t_u_fourth*a_es*np.log(50)  # No emission inside
    I_map[r_grid > 5] = (g**3.7) * t_u_fourth * a_es*(10/3)*(500**0.3 - 10**0.3)

    num_pd = 1 + 16.3*cos_zeta_prime + 6.2*cos_zeta_prime**2
    den_pd = 1 + 1.3*cos_zeta_prime + 4.4*cos_zeta_prime**2
    pd = 0.064*(1 - cos_zeta_prime) * (num_pd/den_pd)

    a_tilde   = (1 - cos_alpha*cos_psi)/(cos_alpha - cos_psi)
    tan_chi_gr = (np.cos(i)*np.sin(phi_grid)) / (a_tilde*np.sin(i) + np.cos(phi_grid))
    chi_gr     = np.arctan(tan_chi_gr)
    tan_chi_sr = -beta*cos_alpha*cos_zeta/(sin_zeta**2 - beta*cos_xi)
    chi_sr     = np.arctan(tan_chi_sr)
    chi_tot    = chi_gr + chi_sr
    pa         = chi_tot

    # Raw sky coords & rotate 90° CCW
    b       = (r_grid/np.sqrt(1 - u))*sin_alpha
    cos_Phi = -np.cos(i)*np.cos(phi_grid)/sin_psi
    sin_Phi = -np.sin(phi_grid)/sin_psi
    X_raw   = b*cos_Phi
    Y_raw   = b*sin_Phi
    X = -Y_raw
    Y =  X_raw

    I_norm = I_map / np.max(I_map)
    # Plot intensity with pcolormesh
    im = ax.pcolormesh(
        X, Y, I_norm,
        cmap='afmhot',
        norm=PowerNorm(gamma=0.4,vmin=1e-3, vmax=1),
        shading='auto',
    )

    for r_c in np.arange(1, 16, 2):
        u_c = 1.0/r_c
        cos_psi_c = np.sin(i) * np.cos(phi)
        y_c = 1 - cos_psi_c
        log_term_c = np.log1p(-y_c / 2)
        cos_alpha_c = 1-y_c*(1-u_c)*(1+(u_c**2*y_c**2)/112-(np.e*u_c*y_c/100)*(log_term_c+y_c/2))
        sin_alpha_c = np.sqrt(1-cos_alpha_c**2)
        b_c = (r_c/np.sqrt(1-u_c)) * sin_alpha_c
        sin_psi_c = np.sqrt(1-cos_psi_c**2)
        cos_Phi_c = -np.cos(i)*np.cos(phi)/sin_psi_c
        sin_Phi_c = -np.sin(phi)/sin_psi_c
        ax.plot(-b_c * sin_Phi_c, b_c * cos_Phi_c, 'w--', linewidth=0.8, alpha=0.7)

    for phi_c in np.deg2rad(np.arange(0, 360, 30)):
        u_c = 1.0/r
        cos_psi_c = np.sin(i) * np.cos(phi_c)
        y_c = 1 - cos_psi_c
        log_term_c = np.log1p(-y_c / 2)
        cos_alpha_c = 1-y_c*(1-u_c)*(1+(u_c**2*y_c**2)/112-(np.e*u_c*y_c/100)*(log_term_c+y_c/2))
        sin_alpha_c = np.sqrt(1-cos_alpha_c**2)
        b_c = (r/np.sqrt(1-u_c)) * sin_alpha_c
        sin_psi_c = np.sqrt(1-cos_psi_c**2)
        cos_Phi_c = -np.cos(i)*np.cos(phi_c)/sin_psi_c
        sin_Phi_c = -np.sin(phi_c)/sin_psi_c
        ax.plot(-b_c * sin_Phi_c, b_c * cos_Phi_c, 'w--', linewidth=0.8, alpha=0.7)

    # # Quiver
    # skip_r = 35
    # skip_phi = 50
    # Xq = X[::skip_r, ::skip_phi]
    # Yq = Y[::skip_r, ::skip_phi]
    # pd_q = pd[::skip_r, ::skip_phi]
    # pa_q = pa[::skip_r, ::skip_phi]
    # scale = scale_factor_dic[i_deg]
    # Ur = -pd_q * np.sin(pa_q) * scale
    # Vr =  pd_q * np.cos(pa_q) * scale
    # ax.quiver(
    #     Xq, Yq, Ur, Vr,
    #     color='deepskyblue', headwidth=0, headlength=0,
    #     headaxislength=0, pivot='middle', scale_units='xy', scale=1,
    #     width=0.002, linewidths=0.3
    # )
    
# 1) define your desired radii and azimuths
    r_list   = np.array([1, 3, 5, 7, 10, 15])          # 6 radial rings
    phi_list = np.linspace(0, 2*np.pi, 12, endpoint=False)  # 36 φ values

    # 2) make a small mesh of those points
    Rq, Phiq = np.meshgrid(r_list, phi_list, indexing='xy')

    # 3) compute all the same physics at these Rq,Phiq:
    uq        = 1.0 / Rq
    cos_psiq  = np.sin(i) * np.cos(Phiq)
    sin_psiq  = np.sqrt(1 - cos_psiq**2)
    yq        = 1 - cos_psiq
    log_termq = np.log1p(-yq/2)
    cos_alphaq = 1 - yq*(1-uq)*(1+(uq**2 * yq**2)/112 - (np.e*uq*yq/100)*(log_termq + yq/2))
    sin_alphaq = np.sqrt(np.clip(1 - cos_alphaq**2,0,1))

    betaq  = np.sqrt(uq/(2*(1-uq)))
    gammaq = np.sqrt((1-uq)/(1 - 1.5*uq))

    cos_zetaq       = (sin_alphaq/sin_psiq)*np.cos(i)
    sin_zetaq       = np.sqrt(1 - cos_zetaq**2)
    cos_xiq         = -(sin_alphaq/sin_psiq)*np.sin(i)*np.sin(Phiq)
    deltaq          = 1/(gammaq*(1 - betaq*cos_xiq))
    gq              = deltaq*np.sqrt(1 - uq)
    cos_zetapq      = deltaq*cos_zetaq

    # local polarization
    num_pd = 1 + 16.3*cos_zeta_prime + 6.2*cos_zeta_prime**2
    den_pd = 1 + 1.3*cos_zeta_prime + 4.4*cos_zeta_prime**2
    pd = 0.064*(1 - cos_zeta_prime) * (num_pd/den_pd)
    pd_qmesh        = 0.1171*(1 - cos_zetapq)/(1 + 3.5*cos_zetapq)

    # polarization angle transport
    atan_grq = (np.cos(i)*np.sin(Phiq)) / ((1-cos_alphaq*cos_psiq)/(cos_alphaq-cos_psiq)*np.sin(i) + np.cos(Phiq))
    chi_grq  = np.arctan(atan_grq)
    atan_srq = -betaq*cos_alphaq*cos_zetaq/(sin_zetaq**2 - betaq*cos_xiq)
    chi_srq  = np.arctan(atan_srq)
    pa_qmesh = np.pi/2 + (chi_grq + chi_srq)

    # 4) compute the corresponding sky coords and rotate
    bq      = (Rq/np.sqrt(1-uq)) * sin_alphaq
    cos_Phiq_s = -np.cos(i)*np.cos(Phiq)/sin_psiq
    sin_Phiq_s = -np.sin(Phiq)/sin_psiq
    X_raw_q    = bq * cos_Phiq_s
    Y_raw_q    = bq * sin_Phiq_s
    Xq         = -Y_raw_q
    Yq         =  X_raw_q

    # 5) now build vector components
    scale = scale_factor_dic[i_deg]
    Uq = -pd_qmesh * np.sin(pa_qmesh) * scale
    Vq =  pd_qmesh * np.cos(pa_qmesh) * scale

    # 6) and plot
    ax.quiver(
        Xq, Yq, Uq, Vq,
        color='deepskyblue', headwidth=0, headlength=0, headaxislength=0,
        pivot='middle', scale_units='xy', scale=1,
        width=0.002, linewidths=0.3
    )

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.text(-15, -15, f'$i={i_deg}°$', color='white', fontsize=16,
            bbox=dict(facecolor='black', alpha=0.6, pad=0.2))
    return im

# Main script
fig, axes = plt.subplots(3,1, figsize=(8,24), facecolor='black')
for ax, i_deg in zip(axes, [30,60,80]):
    im = plot_accretion_disk_image(ax, i_deg)
    ax.axis('off')

cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.04, pad=0.02)
cbar.set_label(r'$\log_{10}(I/I_{\max})$', color='white', fontsize=14)
cbar.ax.tick_params(colors='white', labelsize=12)

plt.show()
