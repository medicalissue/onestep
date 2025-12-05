import matplotlib.pyplot as plt
import re

def parse_log(filename):
    epochs = []
    accuracies = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r"Epoch (\d+) Test Accuracy: ([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
    return epochs, accuracies

def main():
    # Strong Teacher Comparison
    # Hardcoded data from walkthrough.md
    iga_epochs = list(range(1, 16))
    iga_acc = [62.16, 65.31, 71.17, 70.17, 73.01, 74.79, 77.09, 77.83, 78.77, 79.09, 79.15, 79.94, 80.41, 80.54, 80.61]
    
    kd_epochs = list(range(1, 16))
    kd_acc = [61.83, 69.74, 71.59, 74.32, 74.99, 77.09, 76.72, 78.50, 78.91, 79.30, 79.71, 80.06, 80.37, 80.52, 80.55]
    
    prism_epochs, prism_acc = parse_log("debug_log_prism.txt")
    prism_c_epochs, prism_c_acc = parse_log("debug_log_prism_c_strong.txt")
    prism_pp_epochs, prism_pp_acc = parse_log("debug_log_prism_pp_strong.txt")
    prism_d_epochs, prism_d_acc = parse_log("debug_log_prism_d_strong.txt")
    
    plt.figure(figsize=(10, 6))
    plt.plot(iga_epochs, iga_acc, label=f"IGA (Final: {iga_acc[-1]}%)", marker='o')
    plt.plot(kd_epochs, kd_acc, label=f"Standard KD (Final: {kd_acc[-1]}%)", marker='x', linestyle='--')
    plt.plot(prism_epochs, prism_acc, label=f"PRISM (Final: {prism_acc[-1]}%)", marker='s', linestyle='-.')
    plt.plot(prism_c_epochs, prism_c_acc, label=f"PRISM-C (Final: {prism_c_acc[-1]}%)", marker='*', linestyle='-')
    plt.plot(prism_pp_epochs, prism_pp_acc, label=f"PRISM++ (Final: {prism_pp_acc[-1]}%)", marker='D', linestyle='-')
    plt.plot(prism_d_epochs, prism_d_acc, label=f"PRISM-D (Final: {prism_d_acc[-1]}%)", marker='o', linestyle='-')
    
    plt.title("PRISM-D vs PRISM++ vs PRISM-C vs PRISM vs IGA vs Standard KD (Strong Teacher)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("comparison_plot_strong_d.png")
    print("Plot saved to comparison_plot_strong_d.png")

if __name__ == "__main__":
    main()
