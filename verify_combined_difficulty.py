"""
Combined Difficulty Score (k-NN + Confidence) 검증

기존 k-NN only 방식 vs 새로운 Combined 방식 비교:
- k-NN only: 높은 k-NN → 무조건 어려움 (문제: high k-NN + high conf 케이스)
- Combined: k-NN + confidence 결합 → 더 정확한 difficulty 추정
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


def compute_difficulty_score(H_raw, probs, targets, conf_weight=0.5):
    """
    k-NN distance와 confidence를 결합한 difficulty score 계산.

    Cases:
    - k-NN 큼 + conf 낮음 → high difficulty → soft teacher ✓
    - k-NN 큼 + conf 높음 → medium difficulty → sharp teacher (기존에 놓침!)
    - k-NN 작음 + conf 높음 → low difficulty → sharp teacher ✓
    - k-NN 작음 + conf 낮음 → medium difficulty → investigate (이상 케이스)
    """
    B = H_raw.size(0)
    device = H_raw.device

    # 1. k-NN based difficulty (batch-relative normalization)
    H_min_batch = H_raw.min()
    H_max_batch = H_raw.max()
    h_range = H_max_batch - H_min_batch

    if h_range < 1e-6:
        knn_difficulty = torch.full_like(H_raw, 0.5)
    else:
        knn_difficulty = (H_raw - H_min_batch) / (h_range + 1e-8)

    # 2. Confidence based difficulty
    confidence = probs[torch.arange(B, device=device), targets]
    conf_difficulty = 1.0 - confidence

    # 3. Combine (weighted average)
    difficulty = (1 - conf_weight) * knn_difficulty + conf_weight * conf_difficulty

    return difficulty, knn_difficulty, conf_difficulty, confidence


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

    # Collect data with different conf_weights
    all_data = {w: {'knn_diff': [], 'conf_diff': [], 'combined': [],
                    'ce': [], 'correct': [], 'conf': [], 'knn': []}
                for w in [0.0, 0.3, 0.5, 0.7, 1.0]}

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)

            ce = F.cross_entropy(logits, targets, reduction='none')
            correct = (logits.argmax(1) == targets).float()

            for w in all_data.keys():
                combined, knn_diff, conf_diff, conf = compute_difficulty_score(H_raw, probs, targets, conf_weight=w)
                all_data[w]['knn_diff'].extend(knn_diff.cpu().numpy())
                all_data[w]['conf_diff'].extend(conf_diff.cpu().numpy())
                all_data[w]['combined'].extend(combined.cpu().numpy())
                all_data[w]['ce'].extend(ce.cpu().numpy())
                all_data[w]['correct'].extend(correct.cpu().numpy())
                all_data[w]['conf'].extend(conf.cpu().numpy())
                all_data[w]['knn'].extend(epsilon_k.cpu().numpy())

    # Convert to numpy
    for w in all_data.keys():
        for k in all_data[w].keys():
            all_data[w][k] = np.array(all_data[w][k])

    # Use w=0.5 as reference
    ref = all_data[0.5]
    knn = ref['knn']
    conf = ref['conf']
    ce = ref['ce']
    correct = ref['correct']

    # =========================================================
    # Figure 1: Combined Difficulty 분석
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. k-NN only difficulty vs CE (기존)
    ax = axes[0, 0]
    knn_diff = all_data[0.0]['combined']  # w=0 means k-NN only
    r_knn = stats.pearsonr(knn_diff, ce)[0]
    ax.scatter(knn_diff, ce, c=['g' if c else 'r' for c in correct], alpha=0.3, s=5)
    ax.set_xlabel('k-NN Difficulty (w=0)')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'k-NN Only vs CE Loss\nr={r_knn:.3f}')

    # 2. Confidence only difficulty vs CE
    ax = axes[0, 1]
    conf_diff = all_data[1.0]['combined']  # w=1 means conf only
    r_conf = stats.pearsonr(conf_diff, ce)[0]
    ax.scatter(conf_diff, ce, c=['g' if c else 'r' for c in correct], alpha=0.3, s=5)
    ax.set_xlabel('Confidence Difficulty (w=1)')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'Confidence Only vs CE Loss\nr={r_conf:.3f}')

    # 3. Combined difficulty vs CE (w=0.5)
    ax = axes[0, 2]
    combined = all_data[0.5]['combined']
    r_combined = stats.pearsonr(combined, ce)[0]
    ax.scatter(combined, ce, c=['g' if c else 'r' for c in correct], alpha=0.3, s=5)
    ax.set_xlabel('Combined Difficulty (w=0.5)')
    ax.set_ylabel('CE Loss')
    ax.set_title(f'Combined vs CE Loss\nr={r_combined:.3f}')

    # 4. Different conf_weights comparison
    ax = axes[1, 0]
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    correlations = []
    for w in weights:
        r = stats.pearsonr(all_data[w]['combined'], ce)[0]
        correlations.append(r)
    ax.bar(range(len(weights)), correlations, color='steelblue')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f'w={w}' for w in weights])
    ax.set_ylabel('Correlation with CE Loss')
    ax.set_title('conf_weight vs Correlation')
    ax.axhline(y=r_knn, color='r', linestyle='--', label='k-NN only')
    ax.legend()

    # 5. 4 Quadrant Analysis (k-NN vs Confidence)
    ax = axes[1, 1]
    knn_high = knn > np.percentile(knn, 50)
    conf_high = conf > 0.5

    # Quadrants
    q1 = knn_high & conf_high   # High k-NN, High Conf (문제 케이스)
    q2 = ~knn_high & conf_high  # Low k-NN, High Conf (easy)
    q3 = ~knn_high & ~conf_high # Low k-NN, Low Conf (anomaly)
    q4 = knn_high & ~conf_high  # High k-NN, Low Conf (hard)

    colors = np.zeros(len(knn))
    colors[q1] = 0  # Red
    colors[q2] = 1  # Green
    colors[q3] = 2  # Yellow
    colors[q4] = 3  # Blue

    cmap = plt.cm.get_cmap('tab10', 4)
    scatter = ax.scatter(knn, conf, c=colors, cmap=cmap, alpha=0.3, s=5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(np.percentile(knn, 50), color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k-NN Distance')
    ax.set_ylabel('Confidence')
    ax.set_title('4 Quadrant Analysis')

    # Legend
    handles = [plt.scatter([], [], c=cmap(i), s=30) for i in range(4)]
    labels = [f'Q1: High k-NN + High Conf (n={q1.sum()})',
              f'Q2: Low k-NN + High Conf (n={q2.sum()})',
              f'Q3: Low k-NN + Low Conf (n={q3.sum()})',
              f'Q4: High k-NN + Low Conf (n={q4.sum()})']
    ax.legend(handles, labels, loc='upper right', fontsize=7)

    # 6. Quadrant별 CE Loss 분포
    ax = axes[1, 2]
    quadrant_ce = [ce[q1], ce[q2], ce[q3], ce[q4]]
    bp = ax.boxplot(quadrant_ce, labels=['Q1\n(문제)', 'Q2\n(Easy)', 'Q3\n(Anomaly)', 'Q4\n(Hard)'], patch_artist=True)
    colors_box = ['red', 'green', 'yellow', 'blue']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('CE Loss')
    ax.set_title('CE Loss by Quadrant')

    plt.tight_layout()
    plt.savefig('combined_difficulty_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved to combined_difficulty_analysis.png")

    # =========================================================
    # Figure 2: Combined Score 효과 분석
    # =========================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Q1 (문제 케이스) 에서의 difficulty score 변화
    ax = axes2[0, 0]
    q1_knn_diff = all_data[0.0]['combined'][q1]
    q1_combined = all_data[0.5]['combined'][q1]
    q1_conf_diff = all_data[1.0]['combined'][q1]

    ax.hist(q1_knn_diff, bins=30, alpha=0.5, label=f'k-NN only (mean={q1_knn_diff.mean():.2f})', density=True)
    ax.hist(q1_combined, bins=30, alpha=0.5, label=f'Combined (mean={q1_combined.mean():.2f})', density=True)
    ax.set_xlabel('Difficulty Score')
    ax.set_ylabel('Density')
    ax.set_title('Q1 (High k-NN + High Conf) Difficulty Distribution')
    ax.legend()
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. Q4 (Hard) 에서의 difficulty score 변화
    ax = axes2[0, 1]
    q4_knn_diff = all_data[0.0]['combined'][q4]
    q4_combined = all_data[0.5]['combined'][q4]

    ax.hist(q4_knn_diff, bins=30, alpha=0.5, label=f'k-NN only (mean={q4_knn_diff.mean():.2f})', density=True)
    ax.hist(q4_combined, bins=30, alpha=0.5, label=f'Combined (mean={q4_combined.mean():.2f})', density=True)
    ax.set_xlabel('Difficulty Score')
    ax.set_ylabel('Density')
    ax.set_title('Q4 (High k-NN + Low Conf) Difficulty Distribution')
    ax.legend()
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # 3. Difficulty Score vs Accuracy by Quadrant
    ax = axes2[1, 0]
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    masks = [q1, q2, q3, q4]

    x = np.arange(len(quadrants))
    width = 0.25

    knn_acc = [correct[m].mean() * 100 for m in masks]
    combined_acc = [correct[m].mean() * 100 for m in masks]  # same (acc doesn't change)

    knn_diff_mean = [all_data[0.0]['combined'][m].mean() for m in masks]
    combined_diff_mean = [all_data[0.5]['combined'][m].mean() for m in masks]

    ax2_twin = ax.twinx()
    bars1 = ax.bar(x - width/2, knn_diff_mean, width, label='k-NN Difficulty', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, combined_diff_mean, width, label='Combined Difficulty', color='coral', alpha=0.7)
    line = ax2_twin.plot(x, knn_acc, 'go-', label='Accuracy', linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(quadrants)
    ax.set_ylabel('Mean Difficulty Score')
    ax2_twin.set_ylabel('Accuracy (%)')
    ax.set_title('Difficulty Score vs Accuracy by Quadrant')
    ax.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # 4. 문제 해결 검증: Q1에서 Combined가 더 낮은 difficulty를 주는가?
    ax = axes2[1, 1]

    # Q1: High k-NN + High Conf → 실제로는 easy (acc 높음) → difficulty 낮아야 함
    # k-NN only는 높은 difficulty, Combined는 낮은 difficulty 줘야 함

    improvement_q1 = q1_knn_diff.mean() - q1_combined.mean()
    improvement_q4 = q4_knn_diff.mean() - q4_combined.mean()

    data = {
        'Q1 (문제 케이스)': {
            'Accuracy': correct[q1].mean() * 100,
            'k-NN Difficulty': q1_knn_diff.mean(),
            'Combined Difficulty': q1_combined.mean(),
            'Improvement': improvement_q1
        },
        'Q4 (True Hard)': {
            'Accuracy': correct[q4].mean() * 100,
            'k-NN Difficulty': q4_knn_diff.mean(),
            'Combined Difficulty': q4_combined.mean(),
            'Improvement': improvement_q4
        }
    }

    # Table
    cell_text = []
    for k, v in data.items():
        cell_text.append([f"{v['Accuracy']:.1f}%", f"{v['k-NN Difficulty']:.3f}",
                          f"{v['Combined Difficulty']:.3f}", f"{v['Improvement']:.3f}"])

    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=list(data.keys()),
                     colLabels=['Accuracy', 'k-NN Diff', 'Combined Diff', 'Δ (k-NN - Combined)'],
                     loc='center',
                     cellLoc='center')
    table.scale(1.2, 2)
    ax.set_title('Problem Case Analysis\n(Positive Δ = Combined correctly reduces difficulty)', y=0.7)

    plt.tight_layout()
    plt.savefig('combined_difficulty_effect.png', dpi=150, bbox_inches='tight')
    print("Saved to combined_difficulty_effect.png")

    # =========================================================
    # 수치 분석 출력
    # =========================================================
    print("\n" + "="*70)
    print("Combined Difficulty Score 분석 결과")
    print("="*70)

    print("\n[1] CE Loss와의 상관계수:")
    print(f"  k-NN only (w=0.0): r = {stats.pearsonr(all_data[0.0]['combined'], ce)[0]:.4f}")
    print(f"  Combined  (w=0.5): r = {stats.pearsonr(all_data[0.5]['combined'], ce)[0]:.4f}")
    print(f"  Conf only (w=1.0): r = {stats.pearsonr(all_data[1.0]['combined'], ce)[0]:.4f}")

    print("\n[2] Quadrant 분석:")
    print(f"  Q1 (High k-NN + High Conf): n={q1.sum()}, Acc={100*correct[q1].mean():.1f}%")
    print(f"  Q2 (Low k-NN + High Conf):  n={q2.sum()}, Acc={100*correct[q2].mean():.1f}%")
    print(f"  Q3 (Low k-NN + Low Conf):   n={q3.sum()}, Acc={100*correct[q3].mean():.1f}%")
    print(f"  Q4 (High k-NN + Low Conf):  n={q4.sum()}, Acc={100*correct[q4].mean():.1f}%")

    print("\n[3] Q1 문제 케이스 분석 (High k-NN + High Conf):")
    print(f"  실제 Accuracy: {100*correct[q1].mean():.1f}% (높음 = 실제로 easy)")
    print(f"  k-NN Difficulty: {q1_knn_diff.mean():.3f} (높음 = hard로 잘못 판단)")
    print(f"  Combined Difficulty: {q1_combined.mean():.3f}")
    print(f"  → Combined로 인한 감소: {improvement_q1:.3f}")

    print("\n[4] Q4 True Hard 케이스 분석 (High k-NN + Low Conf):")
    print(f"  실제 Accuracy: {100*correct[q4].mean():.1f}% (낮음 = 실제로 hard)")
    print(f"  k-NN Difficulty: {q4_knn_diff.mean():.3f}")
    print(f"  Combined Difficulty: {q4_combined.mean():.3f}")
    print(f"  → Combined로 인한 변화: {improvement_q4:.3f}")

    print("\n[5] 핵심 인사이트:")
    if improvement_q1 > 0:
        print(f"  ✓ Q1 (문제 케이스)에서 Combined가 difficulty를 {improvement_q1:.3f} 감소시킴")
        print(f"    → 불필요하게 soft teacher를 주던 문제 완화")
    else:
        print(f"  ✗ Q1에서 Combined가 오히려 difficulty 증가 (예상 외)")

    if improvement_q4 < 0.1:
        print(f"  ✓ Q4 (True Hard)에서 difficulty 유지됨 (변화량: {improvement_q4:.3f})")
        print(f"    → 실제 어려운 샘플에는 여전히 soft teacher 적용")
    else:
        print(f"  ⚠ Q4에서 difficulty 감소가 큼 - 조정 필요할 수 있음")


if __name__ == '__main__':
    main()
