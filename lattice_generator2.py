
print('import lib')

import torch
print('import lib')
import torch.optim as optim
print('import lib')
import numpy as np
print('import lib')
from itertools import product
print('import lib')
import os
print('start')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
from tqdm import tqdm

D_in = 32
D_out = 16
base_learning_rate = 0.02
epochs = 3000
batch_size = 500
temperature = 10.0
num_repeats = 5  

print(f"D_in={D_in}, D_out={D_out}, Full code size 2^{D_in}")


def benchmark_mse_approx(M, num_test=100000, batch_test_size=128):

    M = M.to(device)
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        try:
            _, _, Vh = torch.linalg.svd(M)
            N = Vh[D_out:]
            M_square = torch.cat([M, N], dim=0)
            M_square_inv = torch.linalg.inv(M_square)
        except torch.linalg.LinAlgError:
            print("Benchmark SVD/inv failed. M might be singular.")
            return float('inf')
            
    
        num_candidates_exp = D_in - D_out
        padding_vectors = torch.tensor(
            list(product([-1, 1], repeat=num_candidates_exp)), 
            dtype=torch.float32, device=device
        )
        num_candidates = padding_vectors.shape[0]

    for start in tqdm(range(0, num_test, batch_test_size)):
        end = min(start + batch_test_size, num_test)
        bs = end - start

        with torch.no_grad():
            z_batch = torch.randn(bs, D_out, device=device)

          
            z_expanded = z_batch.unsqueeze(1).expand(-1, num_candidates, -1)
            paddings_expanded = padding_vectors.unsqueeze(0).expand(bs, -1, -1)
            Y_subset = torch.cat([z_expanded, paddings_expanded], dim=2)
            Y_subset = Y_subset @ M_square_inv.T
            Y_subset = torch.sign(Y_subset)
            
       
            grid_points_subset = Y_subset @ M.T
            z_expanded_for_dist = z_batch.unsqueeze(1)
            dist_sq_matrix_subset = torch.sum((z_expanded_for_dist - grid_points_subset) ** 2, dim=2)
            
 
            _, nn_idx = torch.min(dist_sq_matrix_subset, dim=1)
            
         
            nearest = torch.gather(grid_points_subset, 1, nn_idx.view(-1, 1, 1).expand(-1, 1, D_out)).squeeze(1)

            mse_batch = torch.mean((z_batch - nearest) ** 2).item()
            total_mse += mse_batch * bs
            total_samples += bs

    return total_mse / total_samples


best_mse = float('inf')
best_M = None

for trial in range(num_repeats):
    print(f"\n===== start {trial+1}/{num_repeats} training =====")
    
    M = torch.randn(D_out, D_in, requires_grad=True, device=device)
    optimizer = optim.Adam([M], lr=base_learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_learning_rate/20)

    num_candidates_exp = D_in - D_out
    padding_vectors = torch.tensor(
        list(product([-1, 1], repeat=num_candidates_exp)), 
        dtype=torch.float32, device=device
    )
    num_candidates = padding_vectors.shape[0]

    for epoch in range(epochs):
        optimizer.zero_grad()
        z_batch = torch.randn(batch_size, D_out, device=device)


        with torch.no_grad(): 
            try:
                
                _, _, Vh = torch.linalg.svd(M)
                N = Vh[D_out:]
                
           
                M_square = torch.cat([M, N], dim=0)
                M_square_inv = torch.linalg.inv(M_square)

            except torch.linalg.LinAlgError:
         
                print(f"Trial {trial+1}, Epoch {epoch+1}: SVD/inv failed, skipping batch.")
                continue


            z_expanded = z_batch.unsqueeze(1).expand(-1, num_candidates, -1)
            paddings_expanded = padding_vectors.unsqueeze(0).expand(batch_size, -1, -1)
            Y_subset = torch.cat([z_expanded, paddings_expanded], dim=2)
            
   
            Y_subset = Y_subset @ M_square_inv.T
            Y_subset = torch.sign(Y_subset) # Shape: (batch_size, num_candidates, D_in)

      
        grid_points_subset = Y_subset @ M.T # Shape: (batch_size, num_candidates, D_out)
        z_expanded_for_dist = z_batch.unsqueeze(1) # Shape: (batch_size, 1, D_out)
        

        dist_sq_matrix_subset = torch.sum((z_expanded_for_dist - grid_points_subset) ** 2, dim=2)
        

        log_sum_exp_subset = torch.logsumexp(-temperature * dist_sq_matrix_subset, dim=1)
        softmin_values = -1.0 / temperature * log_sum_exp_subset
        total_loss = torch.mean(softmin_values) / D_out

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 200 == 0 or epoch == epochs - 1:
            print(f"Trial {trial+1}, Epoch {epoch+1}, Loss: {total_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")


    mse_score = benchmark_mse_approx(M.detach())
    print(f"Trial {trial+1} done, Benchmark MSE: {mse_score:.6f}")


    if mse_score < best_mse:
        best_mse = mse_score
        best_M = M.detach().clone()
        print(f"🎯  best_mse: {best_mse:.6f}")


save_dir = "./lattice/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, str(D_in) + "to" + str(D_out) + ".pt")
torch.save(best_M.cpu(), save_path)
print(f"\n✅  Benchmark MSE: {best_mse:.6f}")
print(f"bset M have been saved to : {save_path}")