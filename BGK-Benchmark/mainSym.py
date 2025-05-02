# mainSym.py
import torch
import numpy as np

from data_gen import generate_training_data_bgk, save_data, load_data
from train import train_sym

# 1) generate or load BGK data
print("[main_naive] Generating training data ...")
f_pre_train, f_post_train = generate_training_data_bgk(N_samples=200_000)  # e.g. 200k
save_data('saved_data/my_bgk_data.npz', f_pre_train, f_post_train)

# Generate a test data
f_pre_test, f_post_test = generate_training_data_bgk(N_samples=50_000)
save_data('saved_data/my_bgk_test_data.npz', f_pre_test, f_post_test)


# 2) train
device = 'cpu' # or 'gpu'
sym_net = train_sym(
    f_pre_train, f_post_train,
    epochs=200, batch_size=32, lr=1e-3,
    hidden_size=50,
    device=device,
    save_path='sym_model.pt'
)
