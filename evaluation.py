import numpy as np
import torch
from tqdm import tqdm

def k_reciprocal_re_ranking(original_dist, k1=20, k2=6, lambda_value=0.3):
    print("Applying k-reciprocal re-ranking...")
    N = original_dist.shape[0]
    
    max_dist = np.max(original_dist, axis=0)
    max_dist[max_dist == 0] = 1.0 
    
    original_dist = original_dist / max_dist
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    
    for i in range(N):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
                
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        
        V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-12) 
        
    jaccard_dist = 1 - V
    final_dist = original_dist * lambda_value + jaccard_dist * (1 - lambda_value)
    return final_dist

def extract_test_features(model, loader, device):
    model.eval()
    feats, pids, cams = [], [], []
    with torch.no_grad():
        for imgs, _, cam_ids, person_ids in tqdm(loader, desc="Extracting Gallery/Query Features"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                feat = model(imgs) 
            feats.append(feat.cpu())
            pids.extend(person_ids.numpy())
            cams.extend(cam_ids.numpy())
    return torch.cat(feats, 0).numpy(), np.array(pids), np.array(cams)

def evaluate_reid(model, loader, device, use_reranking=True):
    feats, pids, cams = extract_test_features(model, loader, device)
    
    dist_mat = 1.0 - (feats @ feats.T)
    dist_mat = np.clip(dist_mat, 0.0, None) 
    
    if use_reranking:
        dist_mat = k_reciprocal_re_ranking(dist_mat)
        
    num_samples = len(pids)
    all_cmc, all_ap = [], []
    
    # Standard Re-ID evaluation protocol (cross-camera matching)
    for i in range(num_samples):
        q_pid, q_cam = pids[i], cams[i]
        
        order = np.argsort(dist_mat[i])
        # Remove identical camera matches and invalid IDs from gallery
        remove_mask = (pids[order] == q_pid) & (cams[order] == q_cam) | (pids[order] <= 0)
        keep_mask = ~remove_mask
        
        raw_cmc = (pids[order][keep_mask] == q_pid).astype(np.int32)
        if not np.any(raw_cmc):
            continue
            
        cmc_cumsum = raw_cmc.cumsum()
        precisions = cmc_cumsum / (np.arange(len(raw_cmc)) + 1)
        ap = (precisions * raw_cmc).sum() / raw_cmc.sum()
        all_ap.append(ap)
        
        if raw_cmc[0] == 1:
            all_cmc.append(np.ones(len(raw_cmc)))
        else:
            first_hit = np.where(raw_cmc == 1)[0][0]
            cmc_vec = np.zeros(len(raw_cmc))
            cmc_vec[first_hit:] = 1
            all_cmc.append(cmc_vec)
            
    mAP = np.mean(all_ap) if all_ap else 0.0
    max_len = max(len(c) for c in all_cmc)
    cmc_matrix = np.zeros((len(all_cmc), max_len))
    for idx, c in enumerate(all_cmc):
        cmc_matrix[idx, :len(c)] = c
        if len(c) < max_len:
            cmc_matrix[idx, len(c):] = 1.0
            
    cmc = np.mean(cmc_matrix, axis=0)
    print(f"\n--- Re-ID Evaluation Results ---")
    print(f"Rank-1: {cmc[0]*100:.2f}% | Rank-5: {cmc[4]*100:.2f}% | Rank-10: {cmc[9]*100:.2f}%")
    print(f"mAP: {mAP*100:.2f}%")
    return cmc[0]