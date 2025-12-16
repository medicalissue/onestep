"""
LA-Oracle 더 깊은 검증

추가 질문들:
1. k-NN distance가 단순히 logit magnitude를 반영하는 건 아닌가?
2. z_oracle이 실제 다음 스텝 logits를 잘 예측하는가?
3. τ* scaling이 실제로 의미 있는 teacher를 만드는가?
4. 어려운 샘플에 soft teacher가 정말 도움이 되는가?
5. Memory bank의 stale 문제는 없는가?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import copy

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


def get_loaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def verify_knn_vs_magnitude():
    """
    검증 1: k-NN distance가 단순히 logit magnitude와 상관있는 건 아닌지?

    만약 k-NN distance ≈ logit norm 이면, 우리가 하는 건
    그냥 "confident한 샘플 vs uncertain한 샘플" 구분일 뿐.
    """
    print("\n" + "="*70)
    print("[검증 1] k-NN distance vs Logit magnitude")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20(num_classes=100).to(device)

    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    trainloader, testloader = get_loaders()

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

    # Collect
    all_knn_dist = []
    all_logit_norm = []
    all_z_oracle_norm = []
    all_max_logit = []
    all_entropy = []

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)

            all_knn_dist.extend(epsilon_k.cpu().numpy())
            all_logit_norm.extend(logits.norm(dim=1).cpu().numpy())
            all_z_oracle_norm.extend(z_oracle.norm(dim=1).cpu().numpy())
            all_max_logit.extend(logits.max(dim=1).values.cpu().numpy())
            all_entropy.extend(entropy(probs).cpu().numpy())

    knn_dist = np.array(all_knn_dist)
    logit_norm = np.array(all_logit_norm)
    z_oracle_norm = np.array(all_z_oracle_norm)
    max_logit = np.array(all_max_logit)
    H = np.array(all_entropy)

    corr_norm, p1 = stats.spearmanr(knn_dist, logit_norm)
    corr_z_norm, p2 = stats.spearmanr(knn_dist, z_oracle_norm)
    corr_max, p3 = stats.spearmanr(knn_dist, max_logit)
    corr_H, p4 = stats.spearmanr(knn_dist, H)

    print(f"\nk-NN distance vs...")
    print(f"  Logit L2 norm:     r={corr_norm:.4f} (p={p1:.2e})")
    print(f"  z_oracle L2 norm:  r={corr_z_norm:.4f} (p={p2:.2e})")
    print(f"  Max logit:         r={corr_max:.4f} (p={p3:.2e})")
    print(f"  Shannon entropy:   r={corr_H:.4f} (p={p4:.2e})")

    print(f"\n해석:")
    if abs(corr_norm) > 0.7:
        print(f"  ⚠️ k-NN distance가 logit magnitude와 매우 높은 상관 ({corr_norm:.2f})")
        print(f"     → k-NN이 단순히 'confident vs uncertain' 구분일 가능성")
    else:
        print(f"  ✓ k-NN distance가 logit magnitude와 독립적 (r={corr_norm:.2f})")
        print(f"     → k-NN이 실제 '클래스 내 위치' 정보를 담고 있음")

    return {'corr_norm': corr_norm, 'corr_H': corr_H}


def verify_lookahead_accuracy():
    """
    검증 2: z_oracle = z - η·∇L 이 실제 다음 스텝 logits를 얼마나 잘 예측하는가?

    이게 맞지 않으면 우리의 "oracle"이 oracle이 아님.
    """
    print("\n" + "="*70)
    print("[검증 2] Lookahead logits (z_oracle) 정확도")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20(num_classes=100).to(device)

    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    trainloader, _ = get_loaders(batch_size=64)

    # One batch experiment
    images, targets = next(iter(trainloader))
    images, targets = images.to(device), targets.to(device)

    # Save original state
    model_copy = copy.deepcopy(model)

    # Current forward
    model.train()
    logits, features = model.forward_with_features(images)
    probs = F.softmax(logits, dim=1)

    # Compute z_oracle (predicted next logits)
    z_oracle, delta_z = compute_lookahead_logits(
        logits, probs, targets, features,
        eta=0.05, weight_decay=5e-4
    )

    # Actually take one gradient step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    ce_loss = F.cross_entropy(logits, targets)
    ce_loss.backward()
    optimizer.step()

    # Get actual next logits
    model.eval()
    with torch.no_grad():
        logits_next_actual, _ = model.forward_with_features(images)

    # Compare
    z_oracle_np = z_oracle.detach().cpu().numpy()
    z_actual_np = logits_next_actual.cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    # Per-sample comparison
    mse_oracle = np.mean((z_oracle_np - z_actual_np)**2, axis=1)
    mse_current = np.mean((logits_np - z_actual_np)**2, axis=1)

    # Correlation of predictions
    corr_oracle_actual = []
    corr_current_actual = []
    for i in range(len(z_oracle_np)):
        r1, _ = stats.spearmanr(z_oracle_np[i], z_actual_np[i])
        r2, _ = stats.spearmanr(logits_np[i], z_actual_np[i])
        corr_oracle_actual.append(r1)
        corr_current_actual.append(r2)

    print(f"\nMSE to actual next logits:")
    print(f"  z_oracle:  {mse_oracle.mean():.6f} ± {mse_oracle.std():.6f}")
    print(f"  z_current: {mse_current.mean():.6f} ± {mse_current.std():.6f}")
    print(f"  개선율: {100*(mse_current.mean() - mse_oracle.mean())/mse_current.mean():.1f}%")

    print(f"\nRank correlation to actual next logits:")
    print(f"  z_oracle:  {np.mean(corr_oracle_actual):.4f}")
    print(f"  z_current: {np.mean(corr_current_actual):.4f}")

    # Direction of change
    delta_actual = z_actual_np - logits_np
    delta_predicted = z_oracle_np - logits_np  # = delta_z

    # Cosine similarity of delta
    cos_sim = []
    for i in range(len(delta_actual)):
        norm_a = np.linalg.norm(delta_actual[i])
        norm_p = np.linalg.norm(delta_predicted[i])
        if norm_a > 1e-6 and norm_p > 1e-6:
            cos = np.dot(delta_actual[i], delta_predicted[i]) / (norm_a * norm_p)
            cos_sim.append(cos)

    print(f"\n변화 방향 (delta) cosine similarity:")
    print(f"  Mean: {np.mean(cos_sim):.4f}")
    print(f"  → 1.0에 가까울수록 lookahead가 정확함")

    if np.mean(cos_sim) > 0.5:
        print(f"\n  ✓ Lookahead가 실제 업데이트 방향을 잘 예측함")
    else:
        print(f"\n  ⚠️ Lookahead 예측이 부정확함 - oracle이 실제로 oracle이 아닐 수 있음")

    return {'mse_oracle': mse_oracle.mean(), 'cos_sim': np.mean(cos_sim)}


def verify_tau_star_effect():
    """
    검증 3: τ* scaling이 실제로 의미있는 teacher를 만드는가?

    질문: 어려운 샘플에 soft teacher (높은 τ*)가 정말 좋은 건가?
    아니면 오히려 해로운 건가?
    """
    print("\n" + "="*70)
    print("[검증 3] τ* scaling 효과 분석")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20(num_classes=100).to(device)

    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    trainloader, testloader = get_loaders()

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

    # Analyze τ* effect
    all_tau = []
    all_kl_with_tau = []
    all_kl_without_tau = []
    all_ce = []
    all_correct = []
    all_oracle_correct = []  # Does oracle predict correctly?
    all_oracle_improves = []  # Does oracle have higher prob for true class?

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)
            h_min, h_max = 0.5, np.log(100)
            H_target = map_kl_to_target_entropy(H_raw, H_min=h_min, H_max=h_max, C=100)

            tau_star, q_star = find_tau_star(z_oracle, H_target)

            # q* without tau scaling (tau=1)
            q_no_tau = F.softmax(z_oracle, dim=1)

            # KL divergences
            kl_with_tau = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(probs + 1e-8)), dim=1)
            kl_without_tau = torch.sum(q_no_tau * (torch.log(q_no_tau + 1e-8) - torch.log(probs + 1e-8)), dim=1)

            ce = F.cross_entropy(logits, targets, reduction='none')
            correct = (logits.argmax(dim=1) == targets).float()
            oracle_correct = (q_star.argmax(dim=1) == targets).float()

            # Does oracle give higher prob to true class than student?
            student_true_prob = probs[torch.arange(B), targets]
            oracle_true_prob = q_star[torch.arange(B), targets]
            oracle_improves = (oracle_true_prob > student_true_prob).float()

            all_tau.extend(tau_star.cpu().numpy())
            all_kl_with_tau.extend(kl_with_tau.cpu().numpy())
            all_kl_without_tau.extend(kl_without_tau.cpu().numpy())
            all_ce.extend(ce.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_oracle_correct.extend(oracle_correct.cpu().numpy())
            all_oracle_improves.extend(oracle_improves.cpu().numpy())

    tau = np.array(all_tau)
    kl_with = np.array(all_kl_with_tau)
    kl_without = np.array(all_kl_without_tau)
    ce = np.array(all_ce)
    correct = np.array(all_correct)
    oracle_correct = np.array(all_oracle_correct)
    oracle_improves = np.array(all_oracle_improves)

    print(f"\n전체 통계:")
    print(f"  Student accuracy: {correct.mean():.1%}")
    print(f"  Oracle accuracy:  {oracle_correct.mean():.1%}")
    print(f"  Oracle가 true class에 더 높은 확률 주는 비율: {oracle_improves.mean():.1%}")

    # τ* 구간별 분석
    tau_bins = [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3)]
    print(f"\nτ* 구간별 분석:")
    print(f"{'τ* range':<12} {'Count':>8} {'Student Acc':>12} {'Oracle Acc':>12} {'Oracle Helps':>14}")
    print("-" * 60)

    for low, high in tau_bins:
        mask = (tau >= low) & (tau < high)
        if mask.sum() > 0:
            print(f"[{low:.1f}, {high:.1f})    {mask.sum():>8} {correct[mask].mean():>12.1%} {oracle_correct[mask].mean():>12.1%} {oracle_improves[mask].mean():>14.1%}")

    # 핵심 질문: τ* 높은 샘플(어렵다고 판단)에서 oracle이 도움되는가?
    high_tau_mask = tau > 2.0
    low_tau_mask = tau < 1.5

    print(f"\n핵심 분석:")
    print(f"  높은 τ* (>2.0) 샘플:")
    print(f"    - Student accuracy: {correct[high_tau_mask].mean():.1%}")
    print(f"    - Oracle accuracy: {oracle_correct[high_tau_mask].mean():.1%}")
    print(f"    - Oracle helps: {oracle_improves[high_tau_mask].mean():.1%}")

    print(f"  낮은 τ* (<1.5) 샘플:")
    print(f"    - Student accuracy: {correct[low_tau_mask].mean():.1%}")
    print(f"    - Oracle accuracy: {oracle_correct[low_tau_mask].mean():.1%}")
    print(f"    - Oracle helps: {oracle_improves[low_tau_mask].mean():.1%}")

    # 논리적 모순 케이스
    print(f"\n⚠️ 잠재적 문제 케이스:")

    # Case: Oracle이 틀리는데 Student는 맞춤
    bad_oracle = (correct == 1) & (oracle_correct == 0)
    print(f"  Oracle 틀림 & Student 맞음: {bad_oracle.sum()} ({100*bad_oracle.mean():.1f}%)")
    if bad_oracle.sum() > 0:
        print(f"    → 이 샘플들의 평균 τ*: {tau[bad_oracle].mean():.2f}")

    # Case: Oracle이 오히려 true class 확률 낮춤
    oracle_hurts = (oracle_improves == 0)
    print(f"  Oracle이 true class 확률 낮춤: {oracle_hurts.sum()} ({100*oracle_hurts.mean():.1f}%)")
    if oracle_hurts.sum() > 0:
        print(f"    → 이 샘플들의 평균 τ*: {tau[oracle_hurts].mean():.2f}")
        print(f"    → 이 샘플들의 Student 정답률: {correct[oracle_hurts].mean():.1%}")

    return {
        'oracle_helps_ratio': oracle_improves.mean(),
        'oracle_hurts_high_tau': (oracle_improves[high_tau_mask] == 0).mean() if high_tau_mask.sum() > 0 else 0
    }


def verify_class_center_assumption():
    """
    검증 4: k-NN distance가 정말 "클래스 중심으로부터의 거리"를 반영하는가?

    실제로 클래스 centroid를 계산해서 비교
    """
    print("\n" + "="*70)
    print("[검증 4] k-NN distance vs 클래스 centroid 거리")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20(num_classes=100).to(device)

    ckpt = torch.load('best_student.pt', map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)

    trainloader, testloader = get_loaders()

    # Compute class centroids from training set
    class_logits = {i: [] for i in range(100)}

    model.eval()
    with torch.no_grad():
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            for i, t in enumerate(targets.cpu().numpy()):
                class_logits[t].append(z_oracle[i].cpu().numpy())

    # Compute centroids
    centroids = {}
    for c in range(100):
        if len(class_logits[c]) > 0:
            centroids[c] = np.mean(class_logits[c], axis=0)

    # Build memory bank
    memory_bank = LogitMemoryBank(num_classes=100, queue_size_per_class=128, logit_dim=100, device=device)
    with torch.no_grad():
        for i, (images, targets) in enumerate(trainloader):
            if i >= 100: break
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)
            memory_bank.update(z_oracle, targets)

    # Compare k-NN distance vs centroid distance
    all_knn_dist = []
    all_centroid_dist = []

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(logits, probs, targets, features, eta=0.05, weight_decay=5e-4)

            H_raw, epsilon_k = same_class_knn_entropy_faiss(z_oracle, targets, memory_bank, k=3)

            # Centroid distance
            centroid_dist = []
            for i, t in enumerate(targets.cpu().numpy()):
                if t in centroids:
                    dist = np.linalg.norm(z_oracle[i].cpu().numpy() - centroids[t])
                    centroid_dist.append(dist)
                else:
                    centroid_dist.append(np.nan)

            all_knn_dist.extend(epsilon_k.cpu().numpy())
            all_centroid_dist.extend(centroid_dist)

    knn_dist = np.array(all_knn_dist)
    centroid_dist = np.array(all_centroid_dist)

    # Remove NaN
    valid = ~np.isnan(centroid_dist)
    knn_dist = knn_dist[valid]
    centroid_dist = centroid_dist[valid]

    corr, p = stats.spearmanr(knn_dist, centroid_dist)

    print(f"\nk-NN distance vs Centroid distance:")
    print(f"  Spearman correlation: {corr:.4f} (p={p:.2e})")

    if corr > 0.5:
        print(f"\n  ✓ k-NN distance가 실제로 클래스 중심 거리와 높은 상관 ({corr:.2f})")
        print(f"     → 우리의 가정이 타당함")
    elif corr > 0.3:
        print(f"\n  ~ k-NN distance가 클래스 중심 거리와 중간 상관 ({corr:.2f})")
        print(f"     → 부분적으로 타당하지만 다른 요소도 있음")
    else:
        print(f"\n  ⚠️ k-NN distance가 클래스 중심 거리와 낮은 상관 ({corr:.2f})")
        print(f"     → '클래스 내 위치' 가정이 약할 수 있음")

    return {'corr_knn_centroid': corr}


def main():
    print("="*70)
    print("LA-Oracle 심층 검증")
    print("="*70)

    results = {}

    # 1. k-NN vs magnitude
    r1 = verify_knn_vs_magnitude()
    results.update(r1)

    # 2. Lookahead accuracy
    r2 = verify_lookahead_accuracy()
    results.update(r2)

    # 3. τ* effect
    r3 = verify_tau_star_effect()
    results.update(r3)

    # 4. Class center assumption
    r4 = verify_class_center_assumption()
    results.update(r4)

    # Summary
    print("\n" + "="*70)
    print("종합 평가")
    print("="*70)

    print(f"\n1. k-NN distance 독립성:")
    if abs(results['corr_norm']) < 0.5:
        print(f"   ✓ k-NN이 logit magnitude와 독립적 (r={results['corr_norm']:.2f})")
    else:
        print(f"   ⚠️ k-NN이 logit magnitude에 의존 (r={results['corr_norm']:.2f})")

    print(f"\n2. Lookahead 정확도:")
    if results['cos_sim'] > 0.5:
        print(f"   ✓ Lookahead가 실제 업데이트 방향 예측 (cos={results['cos_sim']:.2f})")
    else:
        print(f"   ⚠️ Lookahead 예측 부정확 (cos={results['cos_sim']:.2f})")

    print(f"\n3. Oracle 효과:")
    if results['oracle_helps_ratio'] > 0.5:
        print(f"   ✓ Oracle이 대부분의 경우 도움됨 ({100*results['oracle_helps_ratio']:.1f}%)")
    else:
        print(f"   ⚠️ Oracle이 절반 이상에서 해로움 ({100*(1-results['oracle_helps_ratio']):.1f}%)")

    print(f"\n4. 클래스 중심 가정:")
    if results['corr_knn_centroid'] > 0.5:
        print(f"   ✓ k-NN이 클래스 중심 거리 반영 (r={results['corr_knn_centroid']:.2f})")
    else:
        print(f"   ~ k-NN과 클래스 중심 거리 상관 낮음 (r={results['corr_knn_centroid']:.2f})")


if __name__ == '__main__':
    main()
