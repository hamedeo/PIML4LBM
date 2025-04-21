"""
benchmark_tgv.py
Run a D2Q9 LBM simulation of a 2‑D Taylor‑Green vortex, it replaces
the standard BGK collision with a trained NaiveCollision NN.
"""

import numpy as np
import torch
import sys

from model import NaiveCollision
from train import load_model          # loads weights from .pt trained file

# Simulation parameters from the booklet
Lx, Ly   = 32, 32          # grid size
tau      = 1               # relaxation time
c_s2     = 1/3
nu       = (tau - 0.5)*c_s2          # kinematic viscosity (Chapman‑Enskog fromulaion)
n_steps  = 2500
u0       = 1e-2            # initial velocity amplitude
device   = 'cpu'           # 'cuda' if GPU trained data is available
NN_PATH  = 'naive_model.pt'

# D2Q9 parameters
c = np.array([[ 0,  0],
              [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
              [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]], dtype=np.int8)
w = np.array([4/9, 
              1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# ------ Functions
def equilibrium(rho, ux, uy):
    feq = w[:,None,None] * rho * (1. + ((c[:,0,None,None]*ux + c[:,1,None,None]*uy) / c_s2)) + 0.5*(c[:,0,None,None]*ux + c[:,1,None,None]*uy) **2 - ((ux**2 + uy**2) / (2*c_s2))
    return feq # shape (9, Ly, Lx)

def stream(f):             # f.shape = (9, Ly, Lx)
    for i,(cx,cy) in enumerate(c):
        f[i] = np.roll(f[i], shift=cx, axis=1)
        f[i] = np.roll(f[i], shift=cy, axis=0)

# ------ Initialise the simulation
x = np.arange(Lx);  y = np.arange(Ly)
X, Y = np.meshgrid(x, y, indexing='xy')

ux =  u0 * np.cos(2.0*np.pi*X/Lx) * np.sin(2.0*np.pi*Y/Ly)
uy = -u0 * np.cos(2.0*np.pi*Y/Ly) * np.sin(2.0*np.pi*X/Lx)
rho = np.ones_like(ux)

f = equilibrium(rho, ux, uy) # == f_eq => shape.f = (9, Ly, Lx)

# ------ Load the trained model
model = load_model(NaiveCollision, NN_PATH, hidden_size=50, device=device)

def collide_nn(f_pre):              # shape.f_pre (9, Ly, Lx)
    ### for debugging purposes ###########
    # print(f"f_pre shape: {f_pre.shape}")
    # sys.exit()
    ######################################
    # (9, Ly, Lx) –– reshape/T ––> (Ly*Lx, 9) –– model ––> (Ly*Lx,9) –– reshape/T ––> (9, Ly, Lx)
    flat = f_pre.reshape(9, -1).T   # flattened to comply with .pt trained data shape.f_pre = (N_samples,9) = (Lx*Ly, 9)
    with torch.no_grad():
        out = model(torch.from_numpy(flat).float().to(device))
    return out.cpu().numpy().T.reshape(9, Ly, Lx)

def collide_bgk(f_pre, rho, ux, uy): # f_post = f_pre - (1/tau)*(f_pre - f_eq)
    feq = equilibrium(rho, ux, uy)
    return f_pre - (1.0 / tau) * (f_pre - feq)

# ------ Taylor–Green Vortex Simulation
avg_u           = []        # store ⟨|u|⟩ vs. time
analytic_decay  = []        # the exact ⟨|u|⟩ decay

for t in range(n_steps):
    # 1) streaming
    stream(f)

    # 2) compute pre‐collision macros
    rho  = np.sum(f, axis=0)
    ux_p = np.sum(f * c[:,0,None,None], axis=0) / rho
    uy_p = np.sum(f * c[:,1,None,None], axis=0) / rho

    # 3) collide (both BGK or NN)
    # f_post_bgk = collide_bgk(f, rho, ux_p, uy_p)
    f_post_nn  = collide_nn(f)

    # max_bgk = np.max(np.abs(f_post_bgk - f))
    max_nn   = np.max(np.abs(f_post_nn  - f))
    # print(f"step {t+1:4d}: rho: {rho}, ux: {ux}, uy: {uy}, max |Δf| BGK = {max_bgk:.3e}")
    print(f"step {t+1:4d}: max |Δf| NN = {max_nn:.3e}")

    f[:] = f_post_nn # for pure‐NN run
    # f[:] = f_post_bgk # for pure‐BGK run

    # 4) recompute macros on f_post
    rho  = np.sum(f, axis=0)
    ux   = np.sum(f * c[:,0,None,None], axis=0) / rho
    uy   = np.sum(f * c[:,1,None,None], axis=0) / rho

    # 5) diagnostics on the **post‑collision** field
    u_mag = np.sqrt(ux**2 + uy**2)
    avg_u.append(np.mean(u_mag))

    # 6) analytic Taylor‑Green Vortex decay
    k = 2.0 * np.pi / Lx
    analytic_decay.append((u0/np.sqrt(2)) * np.exp(-2.0 * nu * k*k * t))

    # Heartbeat
    if (t+1) % 250 == 0:
        print(f"step {t+1:4d}:  ⟨|u|⟩ = {avg_u[-1]:.3e}, exact = {analytic_decay[-1]:.3e}")

    # Save mag_nn, ux, and uy at the 1000th step for later use
    if t == 999:
        ux_1000 = ux.copy()
        uy_1000 = uy.copy()
        print("Saved ux, and uy at step 1000 for later use.")

# ---------------------------------------------------------------- results ----
print("\nFinished!")
print(f"Initial ⟨|u|⟩: {avg_u[0]:.3e}")
print(f"Final   ⟨|u|⟩: {avg_u[-1]:.3e}")
print("Compare last 10 steps vs. analytic:")
for k in range(-10,0):
    print(f"t={n_steps+k:4d}  sim={avg_u[k]:.3e}  exact={analytic_decay[k]:.3e}")
# ----------------------------------------------------------------- plots ----
import matplotlib.pyplot as plt

plt.semilogy(avg_u,          label='NN Naive')
plt.semilogy(analytic_decay, '--',    label='Analytic')
plt.xlabel('time step')
plt.ylabel(r'$\langle |u|\rangle$')
plt.legend()
plt.tight_layout()
plt.savefig('taylor_green_vortex_decay.png', dpi=300)
plt.show()

# =========================================================================== #
#                              FIELD PLOT SECTION                             #
# =========================================================================== #
### choose a snapshot within the simulated range #############################
snapshot_step = 1000                    # pick any 0 ≤ t < n_steps
##############################################################################

print(f"\nCreating field plot at t = {snapshot_step}")

# ---------------------------------------------------------------------------
# 2.  Build analytic TG field at the same t
k    = 2.0 * np.pi / Lx
dec  = np.exp(-2 * nu * k*k * snapshot_step)
ux_an =  u0*np.cos(k*X)*np.sin(k*Y) * dec
uy_an = -u0*np.cos(k*Y)*np.sin(k*X) * dec
mag_an = np.sqrt(ux_an**2 + uy_an**2)

# ---------------------------------------------------------------------------
# 3.  Extract Naive NN field saved in variables ux, uy at snapshot_step

mag_nn = np.sqrt(ux_1000**2 + uy_1000**2)

# ---------------------------------------------------------------------------
# 4.  Helper to add a panel with subsampled stream‑lines
def add_panel(ax, mag, ux_field, uy_field, title,
              flip=False, vmin=0.0, vmax=None):
    im = ax.imshow(mag,
                   cmap='viridis',
                   origin='lower',
                   extent=[0,2*np.pi,0,2*np.pi],
                   vmin=vmin,
                   vmax=(vmax if vmax is not None else mag.max()))

    # 2) build physical coordinate vectors for streamplot
    xs_full = np.linspace(0, 2*np.pi, Lx)
    ys_full = np.linspace(0, 2*np.pi, Ly)

    step = 1
    xs = xs_full[::step]   # length = Lx//step
    ys = ys_full[::step]   # length = Ly//step

    # 3) sub‐sample your velocity field but do NOT transpose
    u_ss = ux_field[::step, ::step]  # shape (Ly/step, Lx/step)
    v_ss = uy_field[::step, ::step]

    # 4) overlay streamlines in the correct direction
    ax.streamplot(xs, ys, u_ss, v_ss,
                  color='white',
                  density=0.5,
                  linewidth=1,
                  arrowstyle='->',
                  arrowsize=1.0)

    ax.set_aspect('equal')       # square axes in physical units
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['0', 'π', '2π'])
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels(['0', 'π', '2π'])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'$|u|$')

# ---------------------------------------------------------------------------
# 5.  Build and save the figure
fig = plt.figure(figsize=(7,3))
add_panel(plt.subplot(1,2,1),
          mag_nn, ux_1000, uy_1000,
          'NN Naive',
          flip=False,
          vmin=4.8e-3,
          vmax=4.9e-3)

add_panel(plt.subplot(1,2,2),
          mag_an, ux_an, uy_an,
          'Analytic')

plt.suptitle(f'Taylor–Green vortex – t = {snapshot_step}', y=1.02)
plt.tight_layout()
plt.savefig('velocity_field_naive_vs_analytic.png', dpi=300)
plt.show()
print("Saved  velocity_field_naive_vs_analytic.png")