# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import SymCollision, NaiveCollision, MSRELoss

def train_sym(f_pre, f_post,
              epochs=200, batch_size=32, lr=1e-3,
              hidden_size=50, device='cpu',
              save_path='sym_model.pt'):
    X = torch.from_numpy(f_pre).float().to(device)   # shape (N,9)
    Y = torch.from_numpy(f_post).float().to(device)  # shape (N,9)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = SymCollision(hidden_size=hidden_size).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = MSRELoss()

    net.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            yp = net(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if ep % 10 == 0:
            print(f"[train_sym] Epoch {ep}/{epochs}, Loss={tot/len(ds):.6e}")

    torch.save(net.state_dict(), save_path)
    print(f"Trained SymCollision saved to {save_path}")
    return net

def train_naive(f_pre, f_post, 
                epochs=200, batch_size=32, lr=1e-3,
                hidden_size=50, device='cpu'):
    """
    Returns a trained NaiveCollision model.
    """
    X = torch.from_numpy(f_pre).float().to(device)
    Y = torch.from_numpy(f_post).float().to(device)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = NaiveCollision(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    lossFunction = MSRELoss()

    network.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred_y = network(batch_x)
            loss = lossFunction(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataset)
        if (ep+1) % 10 == 0:
            print(f"[train_naive] Epoch {ep+1}/{epochs}, Loss={avg_loss:.6f}")

    return network


def evaluate_model(model, f_pre_test, f_post_test, device='cpu'): # Should we do an MSE test?
    """
    Evaluate MSE on a hold-out test set for quick check.
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(f_pre_test).float().to(device)
        y_true = torch.from_numpy(f_post_test).float().to(device)
        y_pred = model(x_t)
        mse_test = torch.mean((y_pred - y_true)**2).item()
    return mse_test


def save_model(model, filename='naive_model.pt'): # Save trained model weights for future reuse.
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model_class, filename='naive_model.pt', hidden_size=50, device='cpu'): # Load model weights from disk. 
    model = model_class(hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

