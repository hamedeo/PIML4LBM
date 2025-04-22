"""
benchmark_tgv.py
Run a D2Q9 LBM simulation of a 2‑D Taylor‑Green vortex, it replaces
the standard BGK collision with a trained NaiveCollision NN or Sym model.
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
snapshot_step = 1000       # pick any 0 ≤ t < n_steps

# D2Q9 parameters
c = np.array([[ 0,  0],
              [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
              [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]], dtype=np.int8)
w = np.array([4/9, 
              1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36], dtype=np.float64)
# Permutation indices for D2Q9
# 0: I, 1: E, 2: N, 3: W, 4: S, 5: NE, 6: NW, 7: SW, 8: SE
I, E, N, W, S, NE, NW, SW, SE = range(9)
# D8 = {I, r, r2, r3, s, rs, r2s, r3s}
d8_perms = [
    np.array([I,  E,  N,  W,  S,  NE, NW, SW, SE]),        # identity
    np.array([I,  N,  W,  S,  E,  NW, SW, SE, NE]),        # r   (90°)
    np.array([I,  W,  S,  E,  N,  SW, SE, NE, NW]),        # r²  (180°)
    np.array([I,  S,  E,  N,  W,  SE, NE, NW, SW]),        # r³  (270°)
    np.array([I,  W,  N,  E,  S,  NW, NE, SE, SW]),        # s   (mirror x)
    np.array([I,  S,  W,  N,  E,  SW, SE, NE, NW]),        # r s
    np.array([I,  E,  S,  W,  N,  SE, NE, NW, SW]),        # r² s
    np.array([I,  N,  E,  S,  W,  NE, NW, SW, SE])         # r³ s
]
n_d8 = len(d8_perms)

# ------ Functions
def equilibrium(rho, ux, uy):
    feq = w[:,None,None] * rho * (1. + ((c[:,0,None,None]*ux + c[:,1,None,None]*uy) / c_s2)) + 0.5*(c[:,0,None,None]*ux + c[:,1,None,None]*uy) **2 - ((ux**2 + uy**2) / (2*c_s2))
    return feq # shape (9, Ly, Lx)

def stream(f):             # f.shape = (9, Ly, Lx)
    for i,(cx,cy) in enumerate(c):
        f[i] = np.roll(f[i], shift=cx, axis=1)
        f[i] = np.roll(f[i], shift=cy, axis=0)

def permute_pop(f_pop, perm):   # permute the populations
    """
    f_pop of shape (9, Ly, Lx)
    populations permuted along first axis with sigma
    """
    return f_pop[perm, :, :]

# ------ Initialise the simulation
x = np.arange(Lx);  y = np.arange(Ly)
X, Y = np.meshgrid(x, y, indexing='xy')

ux =  u0 * np.cos(2.0*np.pi*X/Lx) * np.sin(2.0*np.pi*Y/Ly)
uy = -u0 * np.cos(2.0*np.pi*Y/Ly) * np.sin(2.0*np.pi*X/Lx)
rho = np.ones_like(ux)

f0 = equilibrium(rho, ux, uy)    # for NN Naive
f1 = f0.copy()                   # for NN Sym

# ------ Load the trained model
model = load_model(NaiveCollision, NN_PATH, hidden_size=50, device=device)

#############################################################
# ------ Collision functions
#############################################################
#  Group‑averaged NN collision
# f_post = (1/8) Sum[\phi_NN(\sigma * f_pre)/\sigma]
def collide_nn_sym(f_pre):
    acc = np.zeros_like(f_pre)      # shape.f_pre = (9, Ly, Lx)
    flat_all = []                   # store all 8 permuted copies

    # 1) build the 8 permuted copies
    for perm in d8_perms:   # shape.perm = (9,), perm is \sigma
        fp = permute_pop(f_pre, perm)   # \sigma * f_pre, f_permuted = f_pre[sigma, :, :]
        flat_all.append(fp.reshape(9, -1).T)    # shape.f_pre = (Lx*Ly, 9)

    flat_all = np.concatenate(flat_all, axis=0)  # (8*Ly*Lx, 9)

    # 2) evaluate NN on the batch
    with torch.no_grad():
        out_all = model(torch.from_numpy(flat_all).float().to(device))
    out_all = out_all.cpu().numpy()   # (8*Ly*Lx, 9)

    # 3) un‑permute & accumulate
    n_cells = Ly*Lx
    for k, perm in enumerate(d8_perms):
        out_k = out_all[k*n_cells : (k+1)*n_cells, :].T.reshape(9, Ly, Lx)
        acc += permute_pop(out_k, np.argsort(perm)) # unpermute

    return acc / n_d8

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

#############################################################
# ------ Taylor–Green Vortex Simulation
#############################################################
# store ⟨|u|⟩ vs. time
avg_naive = []
avg_sym   = []
analytic  = []
analytic_decay  = []        # the exact ⟨|u|⟩ decay

for t in range(n_steps):
    # 1) streaming
    stream(f0)
    stream(f1)

    # 2) compute pre‐collision macros
    rho0  = np.sum(f0, axis=0)
    ux0_p = np.sum(f0 * c[:,0,None,None], axis=0) / rho0
    uy0_p = np.sum(f0 * c[:,1,None,None], axis=0) / rho0

    rho1  = np.sum(f1, axis=0)
    ux1_p = np.sum(f1 * c[:,0,None,None], axis=0) / rho1
    uy1_p = np.sum(f1 * c[:,1,None,None], axis=0) / rho1

    # 3) collide BGK, NN Naive, and NN Sym

    # f_post_bgk = collide_bgk(f, rho, ux_p, uy_p)
    f0_post = collide_nn(f0)
    f1_post = collide_nn_sym(f1)    # D8‑equivariant NN

    # max_bgk = np.max(np.abs(f_post_bgk - f))
    max_nn_0 = np.max(np.abs(f0_post  - f0))
    max_nn_1 = np.max(np.abs(f1_post  - f1))
    # print(f"step {t+1:4d}: rho: {rho}, ux: {ux}, uy: {uy}, max |Δf| BGK = {max_bgk:.3e}")
    print(f"step {t+1:4d}: max |Δf| NN = Naive: {max_nn_0:.6e}, Sym: {max_nn_1:.6e}")

    # f[:] = f_post_bgk
    f0[:]   = f0_post
    f1[:]   = f1_post

    # 4) recompute macros on f_post
    rho0  = np.sum(f0, axis=0)
    ux0   = np.sum(f0 * c[:,0,None,None], axis=0) / rho0
    uy0   = np.sum(f0 * c[:,1,None,None], axis=0) / rho0

    rho1  = np.sum(f1, axis=0)
    ux1   = np.sum(f1 * c[:,0,None,None], axis=0) / rho1
    uy1   = np.sum(f1 * c[:,1,None,None], axis=0) / rho1

    # 5) compute the average velocity magnitude
    # ⟨|u|⟩ = (1/Lx*Ly) Sum[ sqrt(ux² + uy²) ]
    avg_naive.append( np.mean(np.sqrt(ux0**2 + uy0**2)) )
    avg_sym.append( np.mean(np.sqrt(ux1**2 + uy1**2)) )

    # 6) analytic Taylor‑Green Vortex decay
    k = 2.0 * np.pi / Lx
    analytic_decay.append((u0/np.sqrt(2)) * np.exp(-2.0 * nu * k*k * t))

    # Heartbeat every 250 steps
    if (t+1) % 250 == 0:
        print(
            f"step {t+1:4d}:  "
            f"NN Naive ⟨|u|⟩={avg_naive[-1]:.3e},  "
            f"NN Sym   ⟨|u|⟩={avg_sym[-1]:.3e},  "
            f"analytic ⟨|u|⟩={analytic_decay[-1]:.3e}"
        )

    # Save ux, and uy at the 1000th step for later use
    if t == snapshot_step-1:
        ux_naive, uy_naive = ux0.copy(), uy0.copy()
        ux_symm,  uy_symm = ux1.copy(), uy1.copy()
        print("Saved ux, and uy at step 1000 for later use.")

# ---------------------------------------------------------------- results ----
print("\nFinished!")

print(f"Initial   Naive ⟨|u|⟩ = {avg_naive[0]:.6e},   Sym ⟨|u|⟩ = {avg_sym[0]:.6e},   analytic ⟨|u|⟩ = {analytic_decay[0]:.6e}")
print(f"Final     Naive ⟨|u|⟩ = {avg_naive[-1]:.6e},   Sym ⟨|u|⟩ = {avg_sym[-1]:.6e},   analytic ⟨|u|⟩ = {analytic_decay[-1]:.6e}")
print("Compare last 10 steps vs. analytic:")
for k in range(-10, 0):
    step = n_steps + k
    print(
        f"t={step:4d}  "
        f"Naive={avg_naive[k]:.6e}  "
        f"Sym  ={avg_sym[k]:.6e}  "
        f"Exact={analytic_decay[k]:.6e}"
    )
# ----------------------------------------------------------------- plots ----
import matplotlib.pyplot as plt

plt.semilogy(avg_naive,          label='NN Naive')
plt.semilogy(avg_sym,   label='NN Sym')
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

print(f"\nCreating field plot at t = {snapshot_step}")

# ---------------------------------------------------------------------------
# 2.  Build analytic TG field at the same t
k    = 2.0 * np.pi / Lx
dec  = np.exp(-2 * nu * k*k * snapshot_step)
ux_an =  u0*np.cos(k*X)*np.sin(k*Y) * dec
uy_an = -u0*np.cos(k*Y)*np.sin(k*X) * dec
mag_an = np.sqrt(ux_an**2 + uy_an**2)

# ---------------------------------------------------------------------------
# 3.  Helper to add a panel with subsampled stream‑lines
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
# 4.  Build and save the figure
fig, axs = plt.subplots(1,3, figsize=(10,3))
add_panel(axs[0], np.sqrt(ux_naive**2+uy_naive**2),
          ux_naive, uy_naive,
          'NN Naive',
          flip=False,
          vmin=4.8e-3,
          vmax=4.9e-3)

add_panel(axs[1], np.sqrt(ux_symm**2 +uy_symm**2),
          ux_symm,  uy_symm,
          'NN Sym')

add_panel(axs[2],
          mag_an, ux_an, uy_an,
          'Analytic')

plt.suptitle(f'Taylor–Green vortex – t = {snapshot_step}', y=1.02)
plt.tight_layout()
plt.savefig('velocity_field_naive_vs_analytic.png', dpi=300)
plt.show()
print("Saved  velocity_field_naive_vs_analytic.png")