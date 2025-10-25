import yaml
import torch
from src.models import Neocognitron
from src.trainers import train_unsupervised
import numpy as np 

with open("src/configs/neocognitron.yaml") as f:
    config = yaml.safe_load(f)

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    net = Neocognitron(config, device=Device)

    def make_pattern(kind=0, size=16, shift=(0, 0)):
        img = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2 + shift[0], size // 2 + shift[1]
        if kind == 0: img[max(0,cx-3):min(size,cx+4), max(0,cy-3):min(size,cy+4)] = 1
        elif kind == 1: img[max(0,cx-1):min(size,cx+2), :] = 1
        elif kind == 2:
            for i in range(size):
                j = i + (cx - cy)
                if 0 <= j < size: img[i, j] = 1
        elif kind == 3:
            for i in range(size):
                for j in range(size):
                    if (i-cx)**2 + (j-cy)**2 < 10: img[i, j] = 1
        else:
            img[cx, cy] = 1
        return img

    rng = np.random.RandomState(0)
    patterns = [make_pattern(k % 5, 16, (rng.randint(-3,4), rng.randint(-3,4))) for k in range(5)]
    patterns = np.stack(patterns)[:, None, :, :].astype(np.float32)
    patterns_tensor = torch.from_numpy(patterns).to(Device)

    print("Training unsupervised...")
    train_unsupervised(net, patterns_tensor, n_epochs=3)

    _, out = net.forward(patterns_tensor[0:1])
    print("Final output:", out.shape)
    
    import os
    import matplotlib.pyplot as plt

    # Make sure experiments folder exists
    os.makedirs("experiments/neocognitron", exist_ok=True)

    # Select pattern to visualize
    pattern_idx = 0
    input_img = patterns_tensor[pattern_idx, 0].cpu().numpy()  # (H, W)

    # 1️⃣ Plot input image
    plt.imshow(input_img, cmap='gray')
    plt.title(f'Input Pattern {pattern_idx}')
    plt.axis('off')
    plt.savefig(f"experiments/neocognitron/pattern_{pattern_idx}_input.png")
    plt.close()

    # 2️⃣ Visualize first S-layer activations (spatial feature maps)
    s_out, _ = net.slayers[0].forward(patterns_tensor[pattern_idx:pattern_idx+1])
    s_out = s_out.squeeze().cpu().numpy()  # (n_planes, H, W)

    for plane_idx in range(s_out.shape[0]):
        plt.imshow(s_out[plane_idx], cmap='hot')
        plt.title(f'S-layer 1, Plane {plane_idx}')
        plt.axis('off')
        plt.savefig(f"experiments/neocognitron/pattern_{pattern_idx}_S1_plane_{plane_idx}.png")
        plt.close()

    # 3️⃣ Final C-layer output as bar plot (pattern detection)
    _, final_out = net.forward(patterns_tensor[pattern_idx:pattern_idx+1])
    final_out = final_out.squeeze().cpu().numpy()  # (n_planes, 1, 1) -> (n_planes,)

    plt.bar(range(len(final_out)), final_out)
    plt.title(f'Final C-layer Activation for Pattern {pattern_idx}')
    plt.xlabel('C-plane')
    plt.ylabel('Activation')
    plt.savefig(f"experiments/neocognitron/pattern_{pattern_idx}_C_final.png")
    plt.close()

    # 4️⃣ Highlight strongest C-plane on input (optional)
    strongest_plane = final_out.argmax()
    plt.imshow(input_img, cmap='gray')
    plt.title(f'Pattern {pattern_idx} → C-plane {strongest_plane} activated')
    plt.axis('off')
    plt.savefig(f"experiments/neocognitron/pattern_{pattern_idx}_highlight.png")
    plt.close()

