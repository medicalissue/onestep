"""
Confidence vs k-NN distance 패턴 분석

가설: Confidence는 k-NN과 선형 관계가 아니라 threshold 기반
- k-NN이 특정 값 이하면 → high confidence
- k-NN이 특정 값 이상이면 → 다양한 confidence (high도 low도 있음)
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20(num_classes=100).to(device)
    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    # Build memory bank
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

    # Collect data
    all_knn = []
    all_conf = []
    all_correct = []
    all_ce = []

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)

            conf = probs[torch.arange(B), targets]
            correct = (logits.argmax(1) == targets).float()
            ce = F.cross_entropy(logits, targets, reduction='none')

            all_knn.extend(epsilon_k.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_ce.extend(ce.cpu().numpy())

    knn = np.array(all_knn)
    conf = np.array(all_conf)
    correct = np.array(all_correct)
    ce = np.array(all_ce)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. k-NN vs Confidence (scatter)
    ax = axes[0, 0]
    colors = ['green' if c else 'red' for c in correct]
    ax.scatter(knn, conf, c=colors, alpha=0.3, s=5)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Confidence (True Class)')
    ax.set_title('k-NN vs Confidence\n(Green=Correct, Red=Wrong)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. k-NN bins vs Confidence distribution (boxplot 스타일)
    ax = axes[0, 1]
    knn_percentiles = [0, 20, 40, 60, 80, 100]
    knn_bins = np.percentile(knn, knn_percentiles)

    conf_by_bin = []
    labels = []
    for i in range(len(knn_bins)-1):
        mask = (knn >= knn_bins[i]) & (knn < knn_bins[i+1])
        if i == len(knn_bins)-2:
            mask = (knn >= knn_bins[i]) & (knn <= knn_bins[i+1])
        conf_by_bin.append(conf[mask])
        labels.append(f'{knn_percentiles[i]}-{knn_percentiles[i+1]}%')

    bp = ax.boxplot(conf_by_bin, labels=labels, patch_artist=True)
    ax.set_xlabel('k-NN Distance Percentile')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Distribution by k-NN Bins')

    # 3. k-NN vs Confidence (2D histogram - density)
    ax = axes[0, 2]
    h = ax.hist2d(knn, conf, bins=50, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Confidence')
    ax.set_title('k-NN vs Confidence (Density)')

    # 4. Confidence 구간별 k-NN 분포
    ax = axes[1, 0]
    conf_low = conf < 0.3
    conf_mid = (conf >= 0.3) & (conf < 0.7)
    conf_high = conf >= 0.7

    ax.hist(knn[conf_high], bins=50, alpha=0.5, label=f'High conf (≥0.7): n={conf_high.sum()}', density=True)
    ax.hist(knn[conf_mid], bins=50, alpha=0.5, label=f'Mid conf (0.3-0.7): n={conf_mid.sum()}', density=True)
    ax.hist(knn[conf_low], bins=50, alpha=0.5, label=f'Low conf (<0.3): n={conf_low.sum()}', density=True)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Density')
    ax.set_title('k-NN Distribution by Confidence Level')
    ax.legend()

    # 5. k-NN threshold 분석
    ax = axes[1, 1]
    thresholds = np.percentile(knn, np.arange(0, 101, 5))
    high_conf_ratio = []
    correct_ratio = []

    for thresh in thresholds:
        mask = knn <= thresh
        if mask.sum() > 0:
            high_conf_ratio.append((conf[mask] > 0.7).mean())
            correct_ratio.append(correct[mask].mean())
        else:
            high_conf_ratio.append(0)
            correct_ratio.append(0)

    ax.plot(thresholds, high_conf_ratio, 'b-o', label='High conf ratio (>0.7)', markersize=3)
    ax.plot(thresholds, correct_ratio, 'g-s', label='Accuracy', markersize=3)
    ax.set_xlabel('k-NN Distance Threshold (samples with k-NN ≤ thresh)')
    ax.set_ylabel('Ratio')
    ax.set_title('Cumulative: k-NN Threshold vs High Conf / Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 핵심: k-NN 세로축 분석
    ax = axes[1, 2]

    # k-NN을 여러 threshold로 나눠서 분석
    knn_thresholds = np.percentile(knn, [25, 50, 75])

    for i, thresh in enumerate(knn_thresholds):
        below = knn < thresh
        above = knn >= thresh

        below_conf_mean = conf[below].mean()
        below_conf_std = conf[below].std()
        above_conf_mean = conf[above].mean()
        above_conf_std = conf[above].std()

        ax.errorbar(i*2, below_conf_mean, yerr=below_conf_std, fmt='go', capsize=5, label=f'k-NN<{thresh:.1f}' if i==0 else '')
        ax.errorbar(i*2+0.5, above_conf_mean, yerr=above_conf_std, fmt='rs', capsize=5, label=f'k-NN≥{thresh:.1f}' if i==0 else '')

    ax.set_xticks([0.25, 2.25, 4.25])
    ax.set_xticklabels(['25th %ile', '50th %ile', '75th %ile'])
    ax.set_ylabel('Confidence (mean ± std)')
    ax.set_title('Confidence by k-NN Threshold (Vertical Split)')
    ax.legend(['Below threshold', 'Above threshold'])

    plt.tight_layout()
    plt.savefig('confidence_pattern.png', dpi=150, bbox_inches='tight')
    print("Saved to confidence_pattern.png")

    # 수치 분석
    print("\n" + "="*70)
    print("Confidence 패턴 분석")
    print("="*70)

    # k-NN 구간별 통계
    print("\nk-NN 구간별 Confidence 분포:")
    print(f"{'k-NN Percentile':<20} {'Conf Mean':>10} {'Conf Std':>10} {'High Conf %':>12} {'Accuracy':>10}")
    print("-"*65)

    for i in range(len(knn_bins)-1):
        mask = (knn >= knn_bins[i]) & (knn < knn_bins[i+1])
        if i == len(knn_bins)-2:
            mask = (knn >= knn_bins[i]) & (knn <= knn_bins[i+1])
        if mask.sum() > 0:
            print(f"{labels[i]:<20} {conf[mask].mean():>10.3f} {conf[mask].std():>10.3f} {100*(conf[mask]>0.7).mean():>11.1f}% {100*correct[mask].mean():>9.1f}%")

    # 핵심 인사이트
    print("\n" + "="*70)
    print("핵심 인사이트")
    print("="*70)

    # High k-NN에서 high confidence 비율
    knn_high = knn > np.percentile(knn, 75)
    knn_low = knn < np.percentile(knn, 25)

    print(f"\n높은 k-NN (>75%ile):")
    print(f"  High confidence (>0.7) 비율: {100*(conf[knn_high]>0.7).mean():.1f}%")
    print(f"  Accuracy: {100*correct[knn_high].mean():.1f}%")

    print(f"\n낮은 k-NN (<25%ile):")
    print(f"  High confidence (>0.7) 비율: {100*(conf[knn_low]>0.7).mean():.1f}%")
    print(f"  Accuracy: {100*correct[knn_low].mean():.1f}%")

    # 모순 케이스
    print(f"\n⚠️ 모순 케이스:")
    high_knn_high_conf = knn_high & (conf > 0.9)
    print(f"  높은 k-NN + 높은 conf (>0.9): {high_knn_high_conf.sum()} samples ({100*high_knn_high_conf.mean():.1f}%)")
    print(f"    → 이 샘플들의 정답률: {100*correct[high_knn_high_conf].mean():.1f}%")
    print(f"    → '어렵다'고 판단했지만 실제로 매우 confident하고 대부분 맞춤")


if __name__ == '__main__':
    main()
