# data_gen.py
import numpy as np

def generate_training_data_bgk(
    N_samples=1200000,
    rho_min=0.5, rho_max=2.0,
    u_max=0.03,
    tau=1.0,
    c_s=1.0 / np.sqrt(3.0)
):
    """
    Generates (f_pre, f_post) for the D2Q9 BGK model.
    Returns:
        f_pre_array, f_post_array: shape (N_samples, 9).
    """
    V_d = np.array([
        [ 0,  0],
        [ 1,  0],
        [ 0,  1],
        [-1,  0],
        [ 0, -1],
        [ 1,  1],
        [-1,  1],
        [-1, -1],
        [ 1, -1]
    ], dtype=np.float64)

    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

    f_pre_list, f_post_list = [], []

    for _ in range(N_samples):
        rho   = np.random.uniform(rho_min, rho_max)
        speed = np.random.uniform(0, u_max)
        angle = np.random.uniform(0, 2*np.pi)

        ux = speed * np.cos(angle)
        uy = speed * np.sin(angle)

        # --- compute f_eq
        f_eq = np.zeros(9, dtype=np.float64)
        for i in range(9):
            cu = V_d[i,0]*ux + V_d[i,1]*uy
            f_eq[i] = w[i] * rho * (
                1.0 + cu/(c_s**2)
                + 0.5*(cu/(c_s**2))**2
                - 0.5*((ux**2 + uy**2)/(c_s**2))
            )

        # --- generate small random perturbations
        raw_perturb = np.random.normal(loc=0.0, scale=0.001, size=9)
        raw_perturb -= np.mean(raw_perturb)  # mass conservation shift

        f_pre = f_eq + raw_perturb
        f_post = f_pre - (1.0/tau)*(f_pre - f_eq)

        f_pre_list.append(f_pre)
        f_post_list.append(f_post)

    f_pre_array  = np.array(f_pre_list)
    f_post_array = np.array(f_post_list)

    return f_pre_array, f_post_array


def save_data(filename, f_pre, f_post):
    """Save data as a .npz file for easy reload."""
    np.savez_compressed(filename, f_pre=f_pre, f_post=f_post)
    print(f"Saved data to {filename}")


def load_data(filename):
    """Load the .npz file containing f_pre, f_post."""
    data = np.load(filename)
    return data['f_pre'], data['f_post']

