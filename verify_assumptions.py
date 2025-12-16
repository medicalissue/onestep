"""
LA-Oracle 가정 검증 스크립트

핵심 가정:
1. k-NN distance 큼 → 어려운 샘플 (클래스 내 outlier)
2. 어려운 샘플 → Student가 실제로 어려워함 (CE 높음, confidence 낮음)
3. 따라서 softer teacher (높은 τ*) 제공이 도움됨

검증할 것:
- k-NN distance vs CE loss 상관관계
- k-NN distance vs Student confidence 상관관계
- k-NN distance vs 정답 여부
- 논리적 모순 케이스 찾기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Import from train_la_oracle
import sys
sys.path.insert(0, '/home/junesang/distill')

from models import resnet20
from train_la_oracle import (
    LogitMemoryBank,
    same_class_knn_entropy_faiss,
    compute_lookahead_logits,
    entropy,
    find_tau_star,
    map_kl_to_target_entropy
)
import torchvision
import torchvision.transforms as transforms


def get_cifar100_loader(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


def verify_assumptions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = resnet20(num_classes=100).to(device)
    ckpt_path = 'best_student.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'net' in ckpt:
            model.load_state_dict(ckpt['net'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, using random init")

    # Build memory bank from training set
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    memory_bank = LogitMemoryBank(
        num_classes=100,
        queue_size_per_class=128,
        logit_dim=100,
        device=device
    )

    print("Building memory bank...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(trainloader):
            if batch_idx >= 100:
                break
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(
                logits, probs, targets, features, eta=0.05, weight_decay=5e-4
            )
            memory_bank.update(z_oracle, targets)
    print("Memory bank built")

    # Collect statistics
    testloader = get_cifar100_loader(batch_size=256)

    all_knn_dist = []
    all_ce_loss = []
    all_confidence = []
    all_correct = []
    all_tau_star = []
    all_h_current = []
    all_h_target = []
    all_kl_div = []
    all_margin = []  # logit margin (top1 - top2)

    model.eval()
    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)

            # CE loss per sample
            ce_loss = F.cross_entropy(logits, targets, reduction='none')

            # Confidence (probability of true class)
            confidence = probs[torch.arange(B), targets]

            # Correct prediction
            pred = logits.argmax(dim=1)
            correct = (pred == targets).float()

            # Logit margin
            top2_logits, _ = logits.topk(2, dim=1)
            margin = top2_logits[:, 0] - top2_logits[:, 1]

            # Lookahead logits
            z_oracle, _ = compute_lookahead_logits(
                logits, probs, targets, features, eta=0.05, weight_decay=5e-4
            )

            # k-NN distance (Same-class)
            H_raw, epsilon_k = same_class_knn_entropy_faiss(
                z_oracle, targets, memory_bank, k=3
            )

            # H_target mapping
            h_min, h_max = 0.5, np.log(100)
            H_target = map_kl_to_target_entropy(H_raw, H_min=h_min, H_max=h_max, C=100)

            # tau* and oracle distribution
            tau_star, q_star = find_tau_star(z_oracle, H_target)

            # Current entropy
            H_current = entropy(probs)

            # KL divergence
            kl_div = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(probs + 1e-8)), dim=1)

            # Collect
            all_knn_dist.extend(epsilon_k.cpu().numpy())
            all_ce_loss.extend(ce_loss.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_tau_star.extend(tau_star.cpu().numpy())
            all_h_current.extend(H_current.cpu().numpy())
            all_h_target.extend(H_target.cpu().numpy())
            all_kl_div.extend(kl_div.cpu().numpy())
            all_margin.extend(margin.cpu().numpy())

    # Convert to numpy
    knn_dist = np.array(all_knn_dist)
    ce_loss = np.array(all_ce_loss)
    confidence = np.array(all_confidence)
    correct = np.array(all_correct)
    tau_star = np.array(all_tau_star)
    h_current = np.array(all_h_current)
    h_target = np.array(all_h_target)
    kl_div = np.array(all_kl_div)
    margin = np.array(all_margin)

    print("\n" + "="*70)
    print("가정 검증 결과")
    print("="*70)

    # 1. k-NN distance vs CE loss
    corr_knn_ce, p_knn_ce = stats.spearmanr(knn_dist, ce_loss)
    print(f"\n[1] k-NN distance vs CE loss")
    print(f"    Spearman correlation: {corr_knn_ce:.4f} (p={p_knn_ce:.2e})")
    print(f"    기대: 양의 상관관계 (k-NN 크면 CE도 커야 함)")
    print(f"    결과: {'✓ 기대와 일치' if corr_knn_ce > 0.1 else '✗ 기대와 다름'}")

    # 2. k-NN distance vs Confidence
    corr_knn_conf, p_knn_conf = stats.spearmanr(knn_dist, confidence)
    print(f"\n[2] k-NN distance vs Confidence")
    print(f"    Spearman correlation: {corr_knn_conf:.4f} (p={p_knn_conf:.2e})")
    print(f"    기대: 음의 상관관계 (k-NN 크면 confidence 낮아야 함)")
    print(f"    결과: {'✓ 기대와 일치' if corr_knn_conf < -0.1 else '✗ 기대와 다름'}")

    # 3. k-NN distance vs Correct
    correct_knn_mean = knn_dist[correct == 1].mean()
    wrong_knn_mean = knn_dist[correct == 0].mean()
    print(f"\n[3] k-NN distance vs Correctness")
    print(f"    Correct samples k-NN mean: {correct_knn_mean:.4f}")
    print(f"    Wrong samples k-NN mean: {wrong_knn_mean:.4f}")
    print(f"    기대: Wrong > Correct (틀린 샘플이 k-NN 커야 함)")
    print(f"    결과: {'✓ 기대와 일치' if wrong_knn_mean > correct_knn_mean else '✗ 기대와 다름'}")

    # 4. k-NN distance vs Margin
    corr_knn_margin, p_knn_margin = stats.spearmanr(knn_dist, margin)
    print(f"\n[4] k-NN distance vs Logit Margin")
    print(f"    Spearman correlation: {corr_knn_margin:.4f} (p={p_knn_margin:.2e})")
    print(f"    기대: 음의 상관관계 (k-NN 크면 margin 작아야 함 = 불확실)")
    print(f"    결과: {'✓ 기대와 일치' if corr_knn_margin < -0.1 else '✗ 기대와 다름'}")

    # 5. 논리적 모순 케이스 분석
    print("\n" + "="*70)
    print("논리적 모순 케이스 분석")
    print("="*70)

    # Case A: k-NN 작음 (쉽다고 판단) + CE 높음 (실제로 어려움)
    knn_q25 = np.percentile(knn_dist, 25)
    ce_q75 = np.percentile(ce_loss, 75)
    case_a = (knn_dist < knn_q25) & (ce_loss > ce_q75)
    print(f"\n[Case A] k-NN 작음 + CE 높음 (우리가 쉽다고 했는데 실제론 어려움)")
    print(f"    개수: {case_a.sum()} / {len(knn_dist)} ({100*case_a.mean():.1f}%)")
    if case_a.sum() > 0:
        print(f"    이 샘플들의 평균 confidence: {confidence[case_a].mean():.4f}")
        print(f"    이 샘플들의 정답률: {correct[case_a].mean():.1%}")

    # Case B: k-NN 큼 (어렵다고 판단) + CE 낮음 (실제로 쉬움)
    knn_q75 = np.percentile(knn_dist, 75)
    ce_q25 = np.percentile(ce_loss, 25)
    case_b = (knn_dist > knn_q75) & (ce_loss < ce_q25)
    print(f"\n[Case B] k-NN 큼 + CE 낮음 (우리가 어렵다고 했는데 실제론 쉬움)")
    print(f"    개수: {case_b.sum()} / {len(knn_dist)} ({100*case_b.mean():.1f}%)")
    if case_b.sum() > 0:
        print(f"    이 샘플들의 평균 confidence: {confidence[case_b].mean():.4f}")
        print(f"    이 샘플들의 정답률: {correct[case_b].mean():.1%}")

    # Case C: 정답인데 k-NN 매우 큼
    knn_q90 = np.percentile(knn_dist, 90)
    case_c = (correct == 1) & (knn_dist > knn_q90)
    print(f"\n[Case C] 정답 + k-NN 매우 큼 (정답인데 outlier로 판정)")
    print(f"    개수: {case_c.sum()} / {(correct==1).sum()} ({100*case_c.sum()/(correct==1).sum():.1f}%)")
    if case_c.sum() > 0:
        print(f"    이 샘플들의 평균 CE: {ce_loss[case_c].mean():.4f}")
        print(f"    이 샘플들의 평균 confidence: {confidence[case_c].mean():.4f}")

    # Case D: 오답인데 k-NN 매우 작음
    knn_q10 = np.percentile(knn_dist, 10)
    case_d = (correct == 0) & (knn_dist < knn_q10)
    print(f"\n[Case D] 오답 + k-NN 매우 작음 (틀렸는데 쉽다고 판정)")
    print(f"    개수: {case_d.sum()} / {(correct==0).sum()} ({100*case_d.sum()/max(1,(correct==0).sum()):.1f}%)")
    if case_d.sum() > 0:
        print(f"    이 샘플들의 평균 CE: {ce_loss[case_d].mean():.4f}")
        print(f"    이 샘플들의 평균 confidence: {confidence[case_d].mean():.4f}")

    # 6. 전반적인 평가
    print("\n" + "="*70)
    print("전반적인 평가")
    print("="*70)

    total_contradictions = case_a.sum() + case_b.sum()
    print(f"\n총 모순 케이스: {total_contradictions} / {len(knn_dist)} ({100*total_contradictions/len(knn_dist):.1f}%)")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. k-NN vs CE
    ax = axes[0, 0]
    ax.scatter(knn_dist, ce_loss, alpha=0.3, s=5)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'k-NN vs CE (r={corr_knn_ce:.3f})')
    z = np.polyfit(knn_dist, ce_loss, 1)
    p = np.poly1d(z)
    x_line = np.linspace(knn_dist.min(), knn_dist.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2)

    # 2. k-NN vs Confidence
    ax = axes[0, 1]
    ax.scatter(knn_dist, confidence, alpha=0.3, s=5)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Confidence')
    ax.set_title(f'k-NN vs Confidence (r={corr_knn_conf:.3f})')
    z = np.polyfit(knn_dist, confidence, 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2)

    # 3. k-NN distribution by correctness
    ax = axes[0, 2]
    ax.hist(knn_dist[correct==1], bins=50, alpha=0.5, label=f'Correct (mean={correct_knn_mean:.1f})', density=True)
    ax.hist(knn_dist[correct==0], bins=50, alpha=0.5, label=f'Wrong (mean={wrong_knn_mean:.1f})', density=True)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Density')
    ax.set_title('k-NN Distribution by Correctness')
    ax.legend()

    # 4. Contradiction cases
    ax = axes[1, 0]
    colors = np.zeros(len(knn_dist))
    colors[case_a] = 1  # Red
    colors[case_b] = 2  # Blue
    scatter = ax.scatter(knn_dist, ce_loss, c=colors, cmap='RdYlBu', alpha=0.5, s=10)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'Contradictions: Red=CaseA({case_a.sum()}), Blue=CaseB({case_b.sum()})')
    ax.axvline(knn_q25, color='r', linestyle='--', alpha=0.5)
    ax.axvline(knn_q75, color='b', linestyle='--', alpha=0.5)
    ax.axhline(ce_q25, color='b', linestyle='--', alpha=0.5)
    ax.axhline(ce_q75, color='r', linestyle='--', alpha=0.5)

    # 5. τ* vs CE
    corr_tau_ce, _ = stats.spearmanr(tau_star, ce_loss)
    ax = axes[1, 1]
    ax.scatter(tau_star, ce_loss, alpha=0.3, s=5)
    ax.set_xlabel('τ* (Temperature)')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'τ* vs CE (r={corr_tau_ce:.3f})')

    # 6. H_target vs H_current
    ax = axes[1, 2]
    ax.scatter(h_current, h_target, alpha=0.3, s=5, c=knn_dist, cmap='viridis')
    ax.plot([0, 5], [0, 5], 'k--', alpha=0.5)
    ax.set_xlabel('H_current (Student)')
    ax.set_ylabel('H_target (Oracle)')
    ax.set_title('Current vs Target Entropy')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('assumption_verification.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to assumption_verification.png")

    return {
        'corr_knn_ce': corr_knn_ce,
        'corr_knn_conf': corr_knn_conf,
        'corr_knn_margin': corr_knn_margin,
        'case_a_ratio': case_a.mean(),
        'case_b_ratio': case_b.mean(),
    }


if __name__ == '__main__':
    verify_assumptions()
