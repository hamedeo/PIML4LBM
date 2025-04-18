"""
benchmark_tgv.py
Run a D2Q9 LBM simulation of a 2‑D Taylor‑Green vortex, replacing
the standard BGK collision with a previously‑trained NaiveCollision NN.
"""

import numpy as np
import torch

from model import NaiveCollision
from train import load_model          # loads weights from .pt
# ‑‑ if you want a pure‑BGK reference – import nothing extra!

###############################################################################
# 1.  Simulation parameters
###############################################################################
Lx, Ly   = 32, 32          # grid size  (matches the paper’s small test)
tau      = 1.0             # relaxation time used in training (kept here)
c_s2     = 1.0/3.0         # speed‑of‑sound squared in lattice units
nu       = (tau - 0.5)*c_s2          # kinematic viscosity (Chapman‑Enskog)
n_steps  = 2500
u0       = 1.0e-2          # initial velocity amplitude (well inside ML range)
device   = 'cpu'           # change to 'cuda' if you trained+saved on GPU
NN_PATH  = 'naive_model.pt'   # model you saved in main.py
###############################################################################

# D2Q9 discrete velocities  (cx, cy) and weights
c = np.array([[ 0,  0],
              [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
              [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]], dtype=np.int8)
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# ------------------------------------------------------------------ helpers --
def equilibrium(rho, ux, uy):
    """
    Compute Maxwell‑Boltzmann equilibrium (2nd‑order Hermite) for *vectorised*
    rho, ux, uy arrays of shape (Ly, Lx).
    Returns feq with shape (9, Ly, Lx).
    """
    cu = (c[:,0,None,None]*ux + c[:,1,None,None]*uy) / c_s2
    uu = (ux**2 + uy**2) / (2*c_s2)
    feq = w[:,None,None] * rho * (1. + cu + 0.5*cu**2 - uu)
    return feq

# roll (stream) populations on the lattice
def stream(f):
    # f has shape (9, Ly, Lx)
    for i,(cx,cy) in enumerate(c):
        f[i] = np.roll(f[i], shift=cx, axis=1)   # x‑direction
        f[i] = np.roll(f[i], shift=cy, axis=0)   # y‑direction

# ---------------------------------------------------------------- initialise --
x = np.arange(Lx);  y = np.arange(Ly)
X, Y = np.meshgrid(x, y, indexing='xy')

ux =  u0 * np.cos(2.0*np.pi*X/Lx) * np.sin(2.0*np.pi*Y/Ly)
uy = -u0 * np.cos(2.0*np.pi*Y/Ly) * np.sin(2.0*np.pi*X/Lx)
rho = np.ones_like(ux)

f = equilibrium(rho, ux, uy)              # start from equilibrium

# ---------------------------------------------------------------- load model --
model = load_model(NaiveCollision, NN_PATH, hidden_size=50, device=device)

# we’ll feed the NN batched‑across‑all‑nodes for speed
def collide_nn(f_pre):
    """
    f_pre : np array shape (9, Ly, Lx)
    Returns f_post with same shape, using the neural network.
    """
    flat = f_pre.reshape(9, -1).T          # shape (Ncells, 9)
    with torch.no_grad():
        out = model(torch.from_numpy(flat).float().to(device))
    return out.cpu().numpy().T.reshape(9, Ly, Lx)

def collide_bgk(f_pre, rho, ux, uy):
    """
    Classical BGK collision: f_post = f_pre - (1/tau)*(f_pre - f_eq)
    """
    # compute the same equilibrium you use elsewhere
    feq = equilibrium(rho, ux, uy)
    return f_pre - (1.0 / tau) * (f_pre - feq)

# ---------------------------------------------------------------- simulation --
avg_u           = []        # store ⟨|u|⟩ vs. time
analytic_decay  = []        # the exact ⟨|u|⟩ for comparison

for t in range(n_steps):
    # 1) streaming
    stream(f)

    # 2) compute pre‐collision macros (only needed for collision)
    rho  = np.sum(f, axis=0)
    ux_p = np.sum(f * c[:,0,None,None], axis=0) / rho
    uy_p = np.sum(f * c[:,1,None,None], axis=0) / rho

    # 3) collide (pick either BGK or NN)
    f_post_bgk = collide_bgk(f, rho, ux_p, uy_p)
    f_post_nn  = collide_nn(f)

    max_bgk = np.max(np.abs(f_post_bgk - f))
    max_nn   = np.max(np.abs(f_post_nn  - f))
    print(f"step {t+1:4d}: max |Δf| BGK = {max_bgk:.3e}, NN = {max_nn:.3e}")

    # 3a) for pure‐NN run, uncomment:
    f[:] = f_post_nn

    # or 3b) for pure‐BGK run, uncomment:
    # f[:] = f_post_bgk

    # 4) NOW recompute macroscopic fields on f_post
    rho  = np.sum(f, axis=0)
    ux   = np.sum(f * c[:,0,None,None], axis=0) / rho
    uy   = np.sum(f * c[:,1,None,None], axis=0) / rho

    # 5) diagnostics on the **post‑collision** field
    u_mag = np.sqrt(ux**2 + uy**2)
    avg_u.append(np.mean(u_mag))

    # 6) analytic Taylor‑Green decay (fixed formula!)
    k = 2.0 * np.pi / Lx
    analytic_decay.append((u0/np.sqrt(2)) * np.exp(-nu * k*k * t))

    # optional debug on f‐changes
    if (t+1) % 250 == 0:
        print(f"step {t+1:4d}:  ⟨|u|⟩ = {avg_u[-1]:.3e}, exact = {analytic_decay[-1]:.3e}")

# ---------------------------------------------------------------- results ----
print("\nFinished!")
print(f"Initial ⟨|u|⟩: {avg_u[0]:.3e}")
print(f"Final   ⟨|u|⟩: {avg_u[-1]:.3e}")
print("Compare last 10 steps vs. analytic:")
for k in range(-10,0):
    print(f"t={n_steps+k:4d}  sim={avg_u[k]:.3e}  exact={analytic_decay[k]:.3e}")
# ----------------------------------------------------------------- plots ----
import matplotlib.pyplot as plt

plt.semilogy(avg_u,          label='NN collision')
plt.semilogy(analytic_decay, '--',    label='analytic')
plt.xlabel('time step')
plt.ylabel(r'$\langle |u|\rangle$')
plt.legend()
plt.tight_layout()
plt.show()

