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
    linear_epochs, linear_acc = parse_log("debug_log_ablation_linear.txt")
    exp_epochs, exp_acc = parse_log("debug_log_ablation_exp.txt")
    tanh_epochs, tanh_acc = parse_log("debug_log_ablation_tanh.txt")
    
    plt.figure(figsize=(10, 6))
    plt.plot(linear_epochs, linear_acc, label=f"Linear (Final: {linear_acc[-1]}%)", marker='o', linestyle='-')
    plt.plot(exp_epochs, exp_acc, label=f"Exponential (Final: {exp_acc[-1]}%)", marker='s', linestyle='--')
    plt.plot(tanh_epochs, tanh_acc, label=f"Tanh (Final: {tanh_acc[-1]}%)", marker='^', linestyle='-.')
    
    plt.title("Gamma Function Ablation Study (Weak Teacher)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ablation_plot_gamma.png")
    print("Plot saved to ablation_plot_gamma.png")

if __name__ == "__main__":
    main()

