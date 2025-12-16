"""
Oracle Logit 근사: 어떤 정보로 z_oracle을 가장 잘 예측할 수 있는가?

핵심 질문:
- z_oracle = z - η·Δz 인데, Δz를 다양한 방식으로 근사
- 어떤 feature가 Δz (또는 z_oracle)과 가장 correlate하는가?

Oracle의 본질:
- z_oracle = z⁺ = 한 스텝 후 logits
- Δz = -η·(p - e_y)·(||h||² + 1) - η·λ_wd·z + momentum_term

근사 후보:
1. 단순 CE gradient: Δz ≈ -(p - e_y)
2. Feature-scaled CE: Δz ≈ -(p - e_y)·||h||²
3. Class prototype 방향: z_oracle ≈ prototype[y]
4. k-NN weighted average: z_oracle ≈ weighted avg of same-class samples
5. Centroid + confidence: z_oracle ≈ λ·centroid + (1-λ)·z
6. EMA of past samples
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.insert(0, '/home/junesang/distill')

from models import resnet20
from train_la_oracle import (
    LogitMemoryBank,
    same_class_knn_entropy_faiss,
    compute_lookahead_logits,
    entropy,
)
import torchvision
import torchvision.transforms as transforms


def compute_oracle_approximations(model, dataloader, memory_bank, device):
    """다양한 oracle 근사 방법 테스트"""

    # Class centroids from memory bank
    class_centroids = {}
    for c in range(100):
        samples = memory_bank.get_class_samples(c)
        if samples.size(0) > 0:
            class_centroids[c] = samples.mean(dim=0)

    results = {
        'z_oracle': [],           # Ground truth
        'z_current': [],          # Current logits

        # Approximations
        'approx_ce_grad': [],     # z - η·(p - e_y)
        'approx_scaled_ce': [],   # z - η·(p - e_y)·||h||²
        'approx_centroid': [],    # class centroid
        'approx_knn_avg': [],     # k-NN average
        'approx_blend': [],       # α·centroid + (1-α)·z
        'approx_direction': [],   # z + direction toward centroid

        # Additional info
        'targets': [],
        'correct': [],
        'confidence': [],
    }

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 40:  # Limit for speed
                break

            images, targets = images.to(device), targets.to(device)
            B, C = images.size(0), 100

            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)

            # Ground truth oracle
            z_oracle, delta_z = compute_lookahead_logits(
                logits, probs, targets, features, eta=0.05, weight_decay=5e-4
            )

            # CE gradient: p - e_y
            one_hot = F.one_hot(targets, num_classes=C).float()
            ce_grad = probs - one_hot  # (B, C)

            # Feature norm
            h_norm_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # (B, 1)

            # Approximation 1: Simple CE gradient
            eta = 0.05
            approx_ce = logits - eta * ce_grad

            # Approximation 2: Scaled CE gradient
            approx_scaled = logits - eta * ce_grad * (h_norm_sq + 1)

            # Approximation 3: Class centroid (directly use centroid as oracle)
            approx_centroid = torch.zeros_like(logits)
            for i in range(B):
                c = targets[i].item()
                if c in class_centroids:
                    approx_centroid[i] = class_centroids[c]
                else:
                    approx_centroid[i] = logits[i]

            # Approximation 4: k-NN weighted average
            approx_knn = torch.zeros_like(logits)
            k = 5
            for i in range(B):
                c = targets[i].item()
                samples = memory_bank.get_class_samples(c)
                if samples.size(0) >= k:
                    dists = torch.norm(logits[i:i+1] - samples, dim=1)
                    _, topk_idx = dists.topk(k, largest=False)
                    topk_samples = samples[topk_idx]
                    # Distance-weighted average
                    weights = 1.0 / (dists[topk_idx] + 1e-8)
                    weights = weights / weights.sum()
                    approx_knn[i] = (topk_samples * weights.unsqueeze(1)).sum(dim=0)
                else:
                    approx_knn[i] = logits[i]

            # Approximation 5: Blend of centroid and current
            alpha = 0.3  # How much to move toward centroid
            approx_blend = (1 - alpha) * logits + alpha * approx_centroid

            # Approximation 6: Move in direction of centroid
            for i in range(B):
                c = targets[i].item()
                if c in class_centroids:
                    direction = class_centroids[c] - logits[i]
                    direction = direction / (torch.norm(direction) + 1e-8)
                    step = torch.norm(delta_z[i])  # Use oracle step size
                    results['approx_direction'].append((logits[i] + step * direction).cpu().numpy())
                else:
                    results['approx_direction'].append(logits[i].cpu().numpy())

            # Collect
            results['z_oracle'].extend(z_oracle.cpu().numpy())
            results['z_current'].extend(logits.cpu().numpy())
            results['approx_ce_grad'].extend(approx_ce.cpu().numpy())
            results['approx_scaled_ce'].extend(approx_scaled.cpu().numpy())
            results['approx_centroid'].extend(approx_centroid.cpu().numpy())
            results['approx_knn_avg'].extend(approx_knn.cpu().numpy())
            results['approx_blend'].extend(approx_blend.cpu().numpy())
            results['targets'].extend(targets.cpu().numpy())
            results['correct'].extend((logits.argmax(1) == targets).float().cpu().numpy())
            results['confidence'].extend(probs[torch.arange(B), targets].cpu().numpy())

    # Convert to numpy
    for k in results:
        results[k] = np.array(results[k])

    return results


def evaluate_approximations(results):
    """각 근사 방법 평가"""

    z_oracle = results['z_oracle']
    z_current = results['z_current']
    correct = results['correct']
    confidence = results['confidence']

    approximations = {
        'current (baseline)': results['z_current'],
        'CE gradient': results['approx_ce_grad'],
        'Scaled CE': results['approx_scaled_ce'],
        'Centroid': results['approx_centroid'],
        'k-NN avg': results['approx_knn_avg'],
        'Blend (α=0.3)': results['approx_blend'],
        'Direction': results['approx_direction'],
    }

    print("\n" + "="*80)
    print("ORACLE APPROXIMATION QUALITY")
    print("="*80)
    print(f"{'Method':<25} {'MSE':<12} {'Cosine Sim':<12} {'L2 Dist':<12} {'Rank Corr':<12}")
    print("-"*75)

    results_list = []
    for name, approx in approximations.items():
        # MSE
        mse = np.mean((approx - z_oracle) ** 2)

        # Cosine similarity (per sample, then average)
        cos_sims = []
        for i in range(len(z_oracle)):
            cos = np.dot(approx[i], z_oracle[i]) / (np.linalg.norm(approx[i]) * np.linalg.norm(z_oracle[i]) + 1e-8)
            cos_sims.append(cos)
        cos_sim = np.mean(cos_sims)

        # L2 distance
        l2_dist = np.mean(np.linalg.norm(approx - z_oracle, axis=1))

        # Rank correlation (do argmax predictions match?)
        oracle_pred = z_oracle.argmax(axis=1)
        approx_pred = approx.argmax(axis=1)
        rank_corr = (oracle_pred == approx_pred).mean()

        results_list.append((name, mse, cos_sim, l2_dist, rank_corr))
        print(f"{name:<25} {mse:>10.4f} {cos_sim:>12.4f} {l2_dist:>12.4f} {rank_corr:>12.4f}")

    return results_list


def analyze_per_difficulty(results, results_list):
    """Difficulty별로 어떤 근사가 좋은지 분석"""

    z_oracle = results['z_oracle']
    correct = results['correct']
    confidence = results['confidence']

    approximations = {
        'current': results['z_current'],
        'CE gradient': results['approx_ce_grad'],
        'Scaled CE': results['approx_scaled_ce'],
        'Centroid': results['approx_centroid'],
        'k-NN avg': results['approx_knn_avg'],
        'Blend': results['approx_blend'],
        'Direction': results['approx_direction'],
    }

    print("\n" + "="*80)
    print("APPROXIMATION QUALITY BY DIFFICULTY (Confidence)")
    print("="*80)

    # Split by confidence
    high_conf = confidence > 0.7
    low_conf = confidence < 0.3

    print(f"\n[High Confidence (>{0.7})] n={high_conf.sum()}")
    print(f"{'Method':<20} {'Cosine Sim':<12} {'Rank Match':<12}")
    print("-"*45)

    for name, approx in approximations.items():
        cos_sims = []
        for i in np.where(high_conf)[0]:
            cos = np.dot(approx[i], z_oracle[i]) / (np.linalg.norm(approx[i]) * np.linalg.norm(z_oracle[i]) + 1e-8)
            cos_sims.append(cos)
        cos_sim = np.mean(cos_sims) if cos_sims else 0

        oracle_pred = z_oracle[high_conf].argmax(axis=1)
        approx_pred = approx[high_conf].argmax(axis=1)
        rank_corr = (oracle_pred == approx_pred).mean()

        print(f"{name:<20} {cos_sim:>10.4f} {rank_corr:>12.4f}")

    print(f"\n[Low Confidence (<{0.3})] n={low_conf.sum()}")
    print(f"{'Method':<20} {'Cosine Sim':<12} {'Rank Match':<12}")
    print("-"*45)

    for name, approx in approximations.items():
        cos_sims = []
        for i in np.where(low_conf)[0]:
            cos = np.dot(approx[i], z_oracle[i]) / (np.linalg.norm(approx[i]) * np.linalg.norm(z_oracle[i]) + 1e-8)
            cos_sims.append(cos)
        cos_sim = np.mean(cos_sims) if cos_sims else 0

        oracle_pred = z_oracle[low_conf].argmax(axis=1)
        approx_pred = approx[low_conf].argmax(axis=1)
        rank_corr = (oracle_pred == approx_pred).mean()

        print(f"{name:<20} {cos_sim:>10.4f} {rank_corr:>12.4f}")

    # Wrong samples
    wrong = correct == 0
    print(f"\n[Wrong Samples] n={wrong.sum()}")
    print(f"{'Method':<20} {'Cosine Sim':<12} {'Rank Match':<12}")
    print("-"*45)

    for name, approx in approximations.items():
        cos_sims = []
        for i in np.where(wrong)[0]:
            cos = np.dot(approx[i], z_oracle[i]) / (np.linalg.norm(approx[i]) * np.linalg.norm(z_oracle[i]) + 1e-8)
            cos_sims.append(cos)
        cos_sim = np.mean(cos_sims) if cos_sims else 0

        oracle_pred = z_oracle[wrong].argmax(axis=1)
        approx_pred = approx[wrong].argmax(axis=1)
        rank_corr = (oracle_pred == approx_pred).mean()

        print(f"{name:<20} {cos_sim:>10.4f} {rank_corr:>12.4f}")


def visualize_approximations(results):
    """시각화"""
    z_oracle = results['z_oracle']
    targets = results['targets']
    confidence = results['confidence']

    approximations = {
        'Current': results['z_current'],
        'Scaled CE': results['approx_scaled_ce'],
        'Centroid': results['approx_centroid'],
        'k-NN avg': results['approx_knn_avg'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (name, approx) in enumerate(approximations.items()):
        ax = axes[idx // 2, idx % 2]

        # Cosine similarity per sample
        cos_sims = []
        for i in range(len(z_oracle)):
            cos = np.dot(approx[i], z_oracle[i]) / (np.linalg.norm(approx[i]) * np.linalg.norm(z_oracle[i]) + 1e-8)
            cos_sims.append(cos)
        cos_sims = np.array(cos_sims)

        # Scatter: confidence vs cosine sim
        ax.scatter(confidence, cos_sims, alpha=0.3, s=5)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Cosine Similarity to Oracle')
        ax.set_title(f'{name}\nMean cos={cos_sims.mean():.3f}')
        ax.axhline(y=cos_sims.mean(), color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('oracle_approximation_quality.png', dpi=150, bbox_inches='tight')
    print("\nSaved to oracle_approximation_quality.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = resnet20(num_classes=100).to(device)
    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Build memory bank
    print("Building memory bank...")
    memory_bank = LogitMemoryBank(num_classes=100, queue_size_per_class=128, logit_dim=100, device=device)
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(trainloader):
            if i >= 100: break
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)
            memory_bank.update(z_oracle, targets)

    # Compute approximations
    print("Computing oracle approximations...")
    results = compute_oracle_approximations(model, testloader, memory_bank, device)

    # Evaluate
    results_list = evaluate_approximations(results)

    # Per-difficulty analysis
    analyze_per_difficulty(results, results_list)

    # Visualize
    visualize_approximations(results)

    # ==================================================================
    # Key insight
    # ==================================================================
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("""
    Oracle z_oracle = z - η·Δz 를 근사하는 가장 좋은 방법:

    1. Scaled CE gradient가 이론적으로 정확 (Δz 계산 자체)
    2. 하지만 Centroid/k-NN avg는 "미래 상태"를 더 잘 반영할 수 있음
       - 이미 학습된 샘플들의 logit 분포 = 학습 완료 후 상태
       - 현재 샘플이 "어디로 가야 하는지" 지시

    새로운 아이디어:
    - Difficulty에 따라 다른 oracle 사용?
    - High confidence: Scaled CE (이미 잘 학습됨, 미세 조정만)
    - Low confidence: Centroid/k-NN (더 aggressive하게 prototype으로 끌어당김)
    """)


if __name__ == '__main__':
    main()
