# main.py
import torch
import numpy as np

from data_gen import generate_training_data_bgk, save_data, load_data
from train import train_naive, evaluate_model, save_model, load_model
from model import NaiveCollision

def main_naive():
    # 1) Generate new training data
    print("[main_naive] Generating training data ...")
    f_pre_train, f_post_train = generate_training_data_bgk(N_samples=200000)  # e.g. 200k
    save_data('saved_data/my_bgk_data.npz', f_pre_train, f_post_train)

    # We can also generate a test set
    f_pre_test, f_post_test = generate_training_data_bgk(N_samples=50000)
    save_data('saved_data/my_bgk_test_data.npz', f_pre_test, f_post_test)

    # 2) Train model
    print("[main_naive] Start training ...")
    device = 'cpu'  # or 'cuda'
    model = train_naive(
        f_pre_train, f_post_train, 
        epochs=200, 
        batch_size=32,
        lr=1e-3,
        hidden_size=50,
        device=device
    )

    # 3) Evaluate on test set
    mse_test = evaluate_model(model, f_pre_test, f_post_test, device=device)
    print(f"[main_naive] Test MSE = {mse_test:.6e}")

    # 4) Save model
    save_model(model, filename='naive_model.pt')


def load_and_evaluate_model():
    print("[main_naive] Loading previously saved data and model to do a quick check.")
    # Load data
    f_pre_test, f_post_test = load_data('saved_data/my_bgk_test_data.npz')

    # Load model
    model = load_model(NaiveCollision, 'naive_model.pt')

    # Evaluate again
    mse_test = evaluate_model(model, f_pre_test, f_post_test)
    print(f"[load_and_evaluate_model] MSE on test data = {mse_test:.6e}")


if __name__=="__main__":
    main_naive()       # Comment out in case of only loading a pre-trained model
    load_and_evaluate_model()

