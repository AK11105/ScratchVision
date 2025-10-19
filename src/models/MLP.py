import torch
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size=784, hidden1_size=256, hidden2_size=128, hidden3_size=64, num_classes=10):
        super(MultiLayerPerceptron, self).__init__()

        #Network Layer
        self.flatten = nn.Flatten() #Flatten 2D/3D image to 1D tensor
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        self.output = nn.Linear(hidden3_size, num_classes)

        #Activation Function
        self.relu = nn.ReLU()

        #Droput for regularization
        self.dropout = nn.Dropout(p=0.25)

         #Store Architecture Info
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.num_classes = num_classes

        '''
        Alternative Formulation:
        
        self.model = nn.Sequential(
            nn.Flatten(),                     # 1×28×28 → 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )
        
        and use it in forward pass like
        
        return self.model(X) '''

    def forward(self, X):
        #X: Input tensor of shape (batch_size, 1, 28, 28)
        #returns: logits: Output tensor of shape (batch_size, num_classes)

        X = self.flatten(X)

        #Hidden Layer 1: 784 -> 256
        X = self.hidden1(X)
        X = self.relu(X)
        X = self.dropout(X)

        #Hidden Layer 2: 256 -> 128
        X = self.hidden2(X)
        X = self.relu(X)
        X = self.dropout(X)

        #Hidden Layer 3: 128 -> 64
        X = self.hidden3(X)
        X = self.relu(X)
        X = self.dropout(X)

        #Output Logits: 64 -> 10
        logits = self.output(X)

        return logits

    def info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = {
            'Architecture': f'{self.input_size} -> {self.hidden1_size} -> {self.hidden2_size} -> {self.num_classes}',
            'Total Parameters': f'{total_params:,}',
            'Trainable Parameters': f'{trainable_params:,}',
            'Model Size (MB)': f'{total_params * 4 / (1024**2):.2f}',  # Assuming float32
        }
        return info

    def visualize(model):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
        # Extract hidden layers
        hidden_layers = [
            module for name, module in model.named_children()
            if isinstance(module, torch.nn.Linear) and name != "output"
        ]
        num_hidden = len(hidden_layers)
    
        # Define layer sizes
        input_size = getattr(model, "input_size", None)
        num_classes = getattr(model, "num_classes", None)
    
        if input_size is None or num_classes is None:
            raise ValueError("Model must have 'input_size' and 'num_classes' attributes")
    
        layers = [f"Input\n({input_size})"]
        layer_sizes = [input_size]
        positions = [0]
    
        for i in range(1, num_hidden + 1):
            layer_size = getattr(model, f"hidden{i}_size", None)
            if layer_size:
                layers.append(f"Hidden {i}\n({layer_size})")
                layer_sizes.append(layer_size)
                positions.append(i)
    
        layers.append(f"Output\n({num_classes})")
        layer_sizes.append(num_classes)
        positions.append(num_hidden + 1)
    
        # Plot as rectangles
        for i, (pos, size, label) in enumerate(zip(positions, layer_sizes, layers)):
            height = np.log10(size + 1) * 0.5
            rect = plt.Rectangle(
                (pos - 0.1, -height / 2), 0.2, height,
                facecolor='lightblue', edgecolor='black', linewidth=2
            )
            ax.add_patch(rect)
    
            # Add layer labels
            ax.text(pos, height / 2 + 0.2, label, ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
            # Add activation labels
            if i < len(layers) - 1:  # not for output layer
                activation = 'ReLU' if i < len(layers) - 2 else 'Softmax'
                ax.text(pos + 0.5, 0, activation, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                        fontsize=10)
    
            # Draw connections
            if i < len(layers) - 1:
                next_height = np.log10(layer_sizes[i + 1] + 1) * 0.5
                for j in range(5):
                    y_start = height / 2 - j * (height / 4)
                    y_end = next_height / 2 - j * (next_height / 4)
                    ax.arrow(pos + 0.1, y_start,
                             0.8, y_end - y_start,
                             head_width=0.05, head_length=0.05,
                             fc='gray', ec='gray', alpha=0.6, length_includes_head=True)
    
        # Styling
        ax.set_xlim(-0.5, positions[-1] + 0.5)
        ax.set_ylim(-2, 2)
        ax.set_title('Multi-Layer Perceptron Architecture', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
    
        plt.tight_layout()
        plt.show()
