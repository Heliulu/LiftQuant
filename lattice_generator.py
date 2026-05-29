import torch
import torch.optim as optim
import numpy as np
from itertools import product
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


D_in = 20
D_out = 10
base_learning_rate = 0.01
epochs = 3000
batch_size = 400
temperature = 10.0
num_repeats = 10  

y_vectors_list = list(product([-1, 1], repeat=D_in))
Y = torch.tensor(y_vectors_list, dtype=torch.float32, device=device)
num_grid_points = Y.shape[0]



def benchmark_mse_scalar(M, num_test=1000000, batch_test_size=128):
    M = M.to(device)
    total_mse = 0.0
    total_samples = 0

    for start in tqdm(range(0, num_test, batch_test_size)):
        end = min(start + batch_test_size, num_test)
        bs = end - start

        z_batch = torch.randn(bs, D_out, device=device)
        grid_points = Y @ M.T

        z_sq = (z_batch ** 2).sum(dim=1, keepdim=True)
        x_sq = (grid_points ** 2).sum(dim=1, keepdim=True).T
        dists = z_sq + x_sq - 2 * z_batch @ grid_points.T

        nn_idx = torch.argmin(dists, dim=1)
        nearest = grid_points[nn_idx]

        mse_batch = torch.mean((z_batch - nearest) ** 2).item()
        total_mse += mse_batch * bs
        total_samples += bs

    return total_mse / total_samples


best_mse = float('inf')
best_M = None

for trial in range(num_repeats):
    print(f"\n===== Start {trial+1}/{num_repeats} times Training =====")
   
    M = torch.randn(D_out, D_in, requires_grad=True, device=device)
    optimizer = optim.Adam([M], lr=base_learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = base_learning_rate/10)

    for epoch in range(epochs):
        optimizer.zero_grad()
        z_batch = torch.randn(batch_size, D_out, device=device)
        grid_points = Y @ M.T

        z_sq = (z_batch ** 2).sum(dim=1, keepdim=True)
        x_sq = (grid_points ** 2).sum(dim=1, keepdim=True).T
        dist_sq_matrix = z_sq + x_sq - 2 * z_batch @ grid_points.T

        log_sum_exp = torch.logsumexp(-temperature * dist_sq_matrix, dim=1)
        softmin_values = -1.0 / temperature * log_sum_exp
        total_loss = torch.mean(softmin_values) / D_out

        total_loss.backward()
        optimizer.step()
        scheduler.step()  

        if (epoch + 1) % 200 == 0 or epoch == epochs - 1:
            print(f"Trial {trial+1}, Epoch {epoch+1}, Loss: {total_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

  
    mse_score = benchmark_mse_scalar(M)
    print(f"Trial {trial+1} done, MSE: {mse_score:.6f}")


    if mse_score < best_mse:
        best_mse = mse_score
        best_M = M.detach().clone()
        print(f"🎯 !  best_mse: {best_mse:.6f}")


save_path = "./lattice/" + str(D_in)+"to"+str(D_out) + ".pt"
torch.save(best_M.cpu(), save_path)
print(f"\n✅  Benchmark MSE: {best_mse:.6f}")
print(f"Bset M have saved to: {save_path}")
