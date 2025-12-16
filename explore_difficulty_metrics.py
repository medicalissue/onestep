"""
Difficulty Metric 탐색: 어떤 metric이 "실제 어려움"과 가장 잘 correlate하는가?

Ground truth for "difficulty":
1. CE Loss: 높으면 어려움
2. Correctness: 틀리면 어려움
3. Margin: (top1 - top2) 작으면 어려움

Candidate metrics:
1. k-NN distance (same-class)
2. 1 - Confidence
3. Entropy (Shannon)
4. k-NN + Confidence (combined)
5. Logit magnitude (||z||)
6. Distance to class centroid
7. Lookahead delta (|Δz|)
8. Cross-class k-NN (nearest different class)
9. Margin-based difficulty
10. Gradient norm (||∇CE||)
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


def compute_all_metrics(model, dataloader, memory_bank, device):
    """모든 candidate metric 계산"""

    all_metrics = {
        'ce_loss': [],           # Ground truth 1
        'correct': [],           # Ground truth 2
        'margin': [],            # Ground truth 3: top1 - top2

        # Candidates
        'knn_dist': [],          # k-NN distance (same-class)
        'confidence': [],        # P(true class)
        'shannon_entropy': [],   # H(p)
        'logit_norm': [],        # ||z||
        'delta_z_norm': [],      # ||Δz|| (lookahead magnitude)
        'cross_knn': [],         # nearest different-class distance
        'centroid_dist': [],     # distance to class centroid in memory
        'max_logit': [],         # max(z) - confidence의 다른 표현
        'softmax_gap': [],       # top1_prob - top2_prob
    }

    # First pass: compute class centroids from memory bank
    class_centroids = {}
    for c in range(100):
        samples = memory_bank.get_class_samples(c)
        if samples.size(0) > 0:
            class_centroids[c] = samples.mean(dim=0)

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)

            # Lookahead
            z_oracle, delta_z = compute_lookahead_logits(
                logits, probs, targets, features, eta=0.05, weight_decay=5e-4
            )

            # Ground truths
            ce = F.cross_entropy(logits, targets, reduction='none')
            correct = (logits.argmax(1) == targets).float()

            # Margin: top1 - top2
            top2_vals, _ = logits.topk(2, dim=1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]

            # k-NN (same-class)
            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)

            # Confidence
            confidence = probs[torch.arange(B, device=device), targets]

            # Shannon entropy
            H = entropy(probs)

            # Logit norm
            logit_norm = torch.norm(logits, dim=1)

            # Delta z norm
            delta_z_norm = torch.norm(delta_z, dim=1)

            # Cross-class k-NN: nearest sample from different class
            cross_knn = torch.zeros(B, device=device)
            z_np = z_oracle.detach().cpu().numpy().astype('float32')
            for i in range(B):
                target_class = targets[i].item()
                min_dist = float('inf')
                for c in range(100):
                    if c == target_class:
                        continue
                    samples = memory_bank.get_class_samples(c)
                    if samples.size(0) > 0:
                        dists = torch.norm(z_oracle[i:i+1] - samples, dim=1)
                        min_dist = min(min_dist, dists.min().item())
                cross_knn[i] = min_dist

            # Centroid distance
            centroid_dist = torch.zeros(B, device=device)
            for i in range(B):
                target_class = targets[i].item()
                if target_class in class_centroids:
                    centroid_dist[i] = torch.norm(z_oracle[i] - class_centroids[target_class])

            # Max logit
            max_logit = logits.max(dim=1).values

            # Softmax gap
            top2_probs, _ = probs.topk(2, dim=1)
            softmax_gap = top2_probs[:, 0] - top2_probs[:, 1]

            # Collect
            all_metrics['ce_loss'].extend(ce.cpu().numpy())
            all_metrics['correct'].extend(correct.cpu().numpy())
            all_metrics['margin'].extend(margin.cpu().numpy())
            all_metrics['knn_dist'].extend(epsilon_k.cpu().numpy())
            all_metrics['confidence'].extend(confidence.cpu().numpy())
            all_metrics['shannon_entropy'].extend(H.cpu().numpy())
            all_metrics['logit_norm'].extend(logit_norm.cpu().numpy())
            all_metrics['delta_z_norm'].extend(delta_z_norm.cpu().numpy())
            all_metrics['cross_knn'].extend(cross_knn.cpu().numpy())
            all_metrics['centroid_dist'].extend(centroid_dist.cpu().numpy())
            all_metrics['max_logit'].extend(max_logit.cpu().numpy())
            all_metrics['softmax_gap'].extend(softmax_gap.cpu().numpy())

    # Convert to numpy
    for k in all_metrics:
        all_metrics[k] = np.array(all_metrics[k])

    return all_metrics


def compute_combined_metrics(metrics):
    """다양한 combination 시도"""

    knn = metrics['knn_dist']
    conf = metrics['confidence']
    H = metrics['shannon_entropy']
    cross_knn = metrics['cross_knn']
    centroid = metrics['centroid_dist']
    delta_z = metrics['delta_z_norm']

    # Normalize to [0, 1]
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    knn_n = normalize(knn)
    conf_n = 1 - conf  # high conf = low difficulty
    H_n = normalize(H)
    cross_knn_n = normalize(cross_knn)
    centroid_n = normalize(centroid)
    delta_z_n = normalize(delta_z)

    combined = {}

    # 1. k-NN + confidence (기존)
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        combined[f'knn_conf_w{w}'] = (1-w) * knn_n + w * conf_n

    # 2. k-NN + entropy
    for w in [0.3, 0.5, 0.7]:
        combined[f'knn_entropy_w{w}'] = (1-w) * knn_n + w * H_n

    # 3. Centroid + confidence
    for w in [0.3, 0.5, 0.7]:
        combined[f'centroid_conf_w{w}'] = (1-w) * centroid_n + w * conf_n

    # 4. (k-NN - cross_kNN) ratio: 같은 클래스와 멀고, 다른 클래스와 가까우면 어려움
    # High ratio = 같은 클래스 멀음 / 다른 클래스 가까움 = 어려움
    ratio = knn / (cross_knn + 1e-8)
    combined['knn_ratio'] = normalize(ratio)

    # 5. k-NN ratio + confidence
    for w in [0.3, 0.5, 0.7]:
        combined[f'ratio_conf_w{w}'] = (1-w) * normalize(ratio) + w * conf_n

    # 6. Delta z norm (lookahead magnitude) + confidence
    for w in [0.3, 0.5, 0.7]:
        combined[f'deltaz_conf_w{w}'] = (1-w) * delta_z_n + w * conf_n

    # 7. Multiplicative: k-NN * (1 - conf)
    combined['knn_x_confn'] = knn_n * conf_n

    # 8. Max of k-NN and conf
    combined['max_knn_conf'] = np.maximum(knn_n, conf_n)

    # 9. Geometric mean
    combined['geom_knn_conf'] = np.sqrt(knn_n * conf_n)

    # 10. Entropy only
    combined['entropy_only'] = H_n

    # 11. Centroid only
    combined['centroid_only'] = centroid_n

    # 12. Cross-class distance (inverse = closer to other class = harder)
    combined['cross_knn_inv'] = 1 - cross_knn_n  # closer to other class = harder

    # 13. Triple: k-NN + conf + cross-class
    combined['triple_knn_conf_cross'] = 0.4 * knn_n + 0.4 * conf_n + 0.2 * (1 - cross_knn_n)

    return combined


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
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

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

    # Compute all metrics
    print("Computing metrics on test set...")
    metrics = compute_all_metrics(model, testloader, memory_bank, device)

    # Compute combined metrics
    print("Computing combined metrics...")
    combined = compute_combined_metrics(metrics)

    # Ground truths
    ce = metrics['ce_loss']
    correct = metrics['correct']
    margin = metrics['margin']

    # ==================================================================
    # Correlation analysis
    # ==================================================================
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Which metric best predicts difficulty?")
    print("="*80)

    results = []

    # Base metrics
    base_metrics = ['knn_dist', 'confidence', 'shannon_entropy', 'logit_norm',
                    'delta_z_norm', 'cross_knn', 'centroid_dist', 'max_logit', 'softmax_gap']

    print("\n[1] BASE METRICS vs CE Loss:")
    print(f"{'Metric':<25} {'r (CE)':<12} {'r (Margin)':<12} {'Wrong vs Right':<20}")
    print("-"*70)

    for name in base_metrics:
        vals = metrics[name]
        r_ce = stats.pearsonr(vals, ce)[0]
        r_margin = stats.pearsonr(vals, -margin)[0]  # negative because low margin = hard

        # Wrong vs right mean
        wrong_mean = vals[correct == 0].mean()
        right_mean = vals[correct == 1].mean()

        # For confidence, invert interpretation
        if name == 'confidence' or name == 'softmax_gap' or name == 'max_logit':
            r_ce = -r_ce
            r_margin = -r_margin
            wrong_mean, right_mean = right_mean, wrong_mean

        results.append((name, r_ce, r_margin, wrong_mean, right_mean))
        print(f"{name:<25} {r_ce:>10.4f} {r_margin:>12.4f} W:{wrong_mean:.3f} R:{right_mean:.3f}")

    print("\n[2] COMBINED METRICS vs CE Loss:")
    print(f"{'Metric':<30} {'r (CE)':<12} {'r (Margin)':<12}")
    print("-"*55)

    combined_results = []
    for name, vals in combined.items():
        r_ce = stats.pearsonr(vals, ce)[0]
        r_margin = stats.pearsonr(vals, -margin)[0]
        combined_results.append((name, r_ce, r_margin))
        print(f"{name:<30} {r_ce:>10.4f} {r_margin:>12.4f}")

    # Sort by CE correlation
    print("\n[3] TOP 10 METRICS (sorted by |r| with CE Loss):")
    print("-"*55)

    all_results = []
    for name, r_ce, r_margin, _, _ in results:
        all_results.append((name, r_ce, r_margin))
    for name, r_ce, r_margin in combined_results:
        all_results.append((name, r_ce, r_margin))

    all_results.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (name, r_ce, r_margin) in enumerate(all_results[:15]):
        print(f"{i+1:>2}. {name:<30} r_CE={r_ce:>7.4f}  r_Margin={r_margin:>7.4f}")

    # ==================================================================
    # Visualizations
    # ==================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Top 12 metrics visualization
    top_metrics = all_results[:12]

    for idx, (name, r_ce, r_margin) in enumerate(top_metrics):
        ax = axes[idx // 4, idx % 4]

        if name in metrics:
            vals = metrics[name]
        else:
            vals = combined[name]

        # Scatter with color by correctness
        colors = ['red' if c == 0 else 'green' for c in correct]
        ax.scatter(vals, ce, c=colors, alpha=0.2, s=3)
        ax.set_xlabel(name)
        ax.set_ylabel('CE Loss')
        ax.set_title(f'{name}\nr(CE)={r_ce:.3f}')

    plt.tight_layout()
    plt.savefig('difficulty_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved to difficulty_metrics_comparison.png")

    # ==================================================================
    # Best metric analysis
    # ==================================================================
    best_name, best_r_ce, best_r_margin = all_results[0]
    print("\n" + "="*80)
    print(f"BEST METRIC: {best_name}")
    print(f"  r(CE Loss) = {best_r_ce:.4f}")
    print(f"  r(Margin) = {best_r_margin:.4f}")
    print("="*80)

    # Quadrant analysis for best metric
    if best_name in metrics:
        best_vals = metrics[best_name]
    else:
        best_vals = combined[best_name]

    # Binary: high difficulty vs low difficulty
    median_diff = np.median(best_vals)
    high_diff = best_vals > median_diff

    print(f"\nHigh difficulty (>{median_diff:.3f}):")
    print(f"  Accuracy: {100*correct[high_diff].mean():.1f}%")
    print(f"  Mean CE: {ce[high_diff].mean():.3f}")

    print(f"\nLow difficulty (<={median_diff:.3f}):")
    print(f"  Accuracy: {100*correct[~high_diff].mean():.1f}%")
    print(f"  Mean CE: {ce[~high_diff].mean():.3f}")


if __name__ == '__main__':
    main()
