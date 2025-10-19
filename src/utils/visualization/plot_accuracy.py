import matplotlib.pyplot as plt 
import numpy as np

def plot_accuracy(train_accuracies, test_accuracies, save_path):
    epochs_range = range(1, len(train_accuracies) + 1)
    plt.plot(epochs_range, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(epochs_range, test_accuracies, 'r-s', label='Test Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # dpi=300 gives high-quality image

    plt.show()
    
def plot_accuracy_improvement(test_accuracies, save_path):
    epochs_range = range(1, len(test_accuracies) + 1)
    improvement = np.array(test_accuracies) - test_accuracies[0]
    plt.bar(epochs_range, improvement, alpha=0.7, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title('Test Accuracy Improvement', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
     # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # dpi=300 gives high-quality image

    plt.show()