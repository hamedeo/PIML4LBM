# ───────────── step where you want the snapshot ─────────────
snapshot_step = 1000        # choose any t you like

# -------------------------- 1) run NN up to snapshot_step -------------------
Lx = Ly = 32
tau = 1.0; cs2 = 1./3.; nu = (tau-0.5)*cs2
u0 = 1.0e-2
k  = 2*np.pi/Lx

# (re‑initialize TG vortex field)
X,Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='xy')
ux0 =  u0*np.cos(k*X)*np.sin(k*Y)
uy0 = -u0*np.cos(k*Y)*np.sin(k*X)
rho0 = np.ones_like(ux0)
f = equilibrium(rho0, ux0, uy0)

# load Naive network
nn = load_model(NaiveCollision, 'naive_model.pt',
                hidden_size=50, device='cpu')
def collide_nn(f_pre):
    flat = f_pre.reshape(9,-1).T
    with torch.no_grad():
        out = nn(torch.from_numpy(flat).float()).cpu().numpy()
    return out.T.reshape(9,Ly,Lx)

# run to snapshot_step
for _ in range(snapshot_step):
    # stream
    for i,(cx,cy) in enumerate(c):
        f[i] = np.roll(np.roll(f[i], cx, axis=1), cy, axis=0)
    # collide
    rho  = f.sum(0)
    ux   = (f*c[:,0,None,None]).sum(0)/rho
    uy   = (f*c[:,1,None,None]).sum(0)/rho
    f    = collide_nn(f)

# macroscopic field from Naive
rho_n = f.sum(0)
ux_n  = (f*c[:,0,None,None]).sum(0)/rho_n
uy_n  = (f*c[:,1,None,None]).sum(0)/rho_n
mag_n = np.sqrt(ux_n**2 + uy_n**2)

# -------------------------- 2) analytic field at same time ------------------
t = snapshot_step
decay = np.exp(-nu * k*k * t)
ux_a =  ux0 * decay        # closed‑form TG decay
uy_a =  uy0 * decay
mag_a = np.sqrt(ux_a**2 + uy_a**2)

# -------------------------- 3) make the figure -----------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(7,3))

def add_panel(idx, mag, ux, uy, title):
    ax = plt.subplot(1,2,idx)
    im = ax.imshow(mag, cmap='viridis', origin='lower')
    # thin stream‑lines
    skip = (slice(None,None,2), slice(None,None,2))
    ax.streamplot(np.arange(Lx), np.arange(Ly),
                  ux[skip].T, uy[skip].T,
                  color='w', density=1.2, linewidth=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04,
                 label=r'$|u|$')

add_panel(1, mag_n, ux_n, uy_n, 'NN Naive')
add_panel(2, mag_a, ux_a, uy_a, 'analytic')

plt.suptitle(f'Taylor–Green vortex, t = {snapshot_step}', y=1.02)
plt.tight_layout()
plt.savefig('velocity_field_naive_vs_analytic.png', dpi=300)
plt.show()
