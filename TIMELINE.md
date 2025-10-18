
# 📚 CNN Learning Timeline: From Scratch Implementation

This timeline is designed to guide you through the evolution of Convolutional Neural Networks (CNNs), emphasizing key innovations and providing direct links to foundational research papers.

---

## 1. Classical CNNs

### 🧠 **Neocognitron (1980)**

* **Key Innovation**: Introduced hierarchical feature extraction with convolution and pooling layers.
* **Learning Focus**: Understand early neural network architectures and pattern recognition mechanisms.
* **Paper**: [Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf)

### 🧠 **Time Delay Neural Network (TDNN) (1987)**

* **Key Innovation**: Utilized time-delay arrangements to capture temporal patterns in sequential data.
* **Learning Focus**: Explore the application of neural networks to speech recognition.
* **Paper**: [Phoneme recognition using time-delay neural networks](https://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf)

### 🧠 **LeNet-5 (1998)**

* **Key Innovation**: Demonstrated the effectiveness of CNNs for handwritten digit recognition.
* **Learning Focus**: Implement a simple CNN architecture for image classification.
* **Paper**: [Gradient-Based Learning Applied to Document Recognition](https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

---

## 2. ImageNet Era: Deep CNNs

### 🧠 **AlexNet (2012)**

* **Key Innovation**: Introduced deep CNNs with ReLU activations and GPU acceleration.
* **Learning Focus**: Implement a deep CNN and understand the impact of depth on performance.
* **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### 🧠 **ZFNet (2013)**

* **Key Innovation**: Improved upon AlexNet by visualizing and understanding convolutional networks.
* **Learning Focus**: Learn about filter visualization and network interpretation.
* **Paper**: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

### 🧠 **VGGNet (2014)**

* **Key Innovation**: Utilized very deep networks with small (3x3) convolution filters.
* **Learning Focus**: Implement a deep network and explore the effects of depth on performance.
* **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### 🧠 **GoogLeNet (Inception v1, 2014)**

* **Key Innovation**: Introduced the inception module to capture multi-scale features.
* **Learning Focus**: Implement inception modules and understand multi-scale feature extraction.
* **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

### 🧠 **ResNet (2015)**

* **Key Innovation**: Introduced residual connections to train very deep networks.
* **Learning Focus**: Implement residual blocks and understand their impact on training deep networks.
* **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### 🧠 **Highway Networks (2015)**

* **Key Innovation**: Introduced gating mechanisms to regulate information flow.
* **Learning Focus**: Explore the use of gates in neural networks.
* **Paper**: [Highway Networks](https://arxiv.org/abs/1505.00387)

### 🧠 **DenseNet (2016)**

* **Key Innovation**: Connected each layer to every other layer in a feed-forward fashion.
* **Learning Focus**: Implement dense connections and understand their benefits.
* **Paper**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

---

## 3. Efficient and Mobile CNNs

### 🧠 **MobileNet v1 (2017)**

* **Key Innovation**: Introduced depthwise separable convolutions for efficient computation.
* **Learning Focus**: Implement depthwise separable convolutions and understand their efficiency.
* **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

### 🧠 **Xception (2017)**

* **Key Innovation**: Extended depthwise separable convolutions to the extreme.
* **Learning Focus**: Implement extreme depthwise separable convolutions and explore their impact.
* **Paper**: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### 🧠 **MobileNet v2 (2018)**

* **Key Innovation**: Introduced inverted residuals and linear bottlenecks.
* **Learning Focus**: Implement inverted residuals and understand their role in efficiency.
* **Paper**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### 🧠 **ShuffleNet v1 & v2 (2017-2018)**

* **Key Innovation**: Utilized pointwise group convolutions and channel shuffle operations.
* **Learning Focus**: Implement pointwise group convolutions and channel shuffling.
* **Paper**: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

### 🧠 **MobileNet v3 (2019)**

* **Key Innovation**: Combined neural architecture search with lightweight blocks.
* **Learning Focus**: Implement neural architecture search and lightweight blocks.
* **Paper**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

### 🧠 **EfficientNet (2019)**

* **Key Innovation**: Introduced compound scaling to optimize accuracy and efficiency.
* **Learning Focus**: Implement compound scaling and understand its benefits.
* **Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

---

## 4. Modern CNN Architectures

### 🧠 **RegNet (2020)**

* **Key Innovation**: Systematic design of network architectures using simple building blocks.
* **Learning Focus**: Implement systematic architecture design principles.
* **Paper**: [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

### 🧠 **ConvNeXt (2022)**

* **Key Innovation**: Modernized ResNet with design principles from Vision Transformers.
* **Learning Focus**: Implement modernized ResNet architectures.
* **Paper**: [ConvNeXt: Revisiting Convolutional Architectures with a Vision Transformer Mindset](https://arxiv.org/abs/2201.03545)

### 🧠 **ConvNeXt V2 (2023)**

* **Key Innovation**: Introduced masked autoencoders for self-supervised learning.
* **Learning Focus**: Implement masked autoencoders and understand self-supervised learning.
* **Paper**: [ConvNeXt V2: Co-Designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)

---

## 5. Specialized and Experimental Architectures

### 🧠 **FractalNet (2016)**

* **Key Innovation**: Introduced fractal designs for deep networks.
* **Learning Focus**: Explore alternative deep network designs.
* **Paper**: [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/abs/1605.07648)

### 🧠 **Capsule Networks (2017)**

* **Key Innovation**: Proposed capsules to represent spatial hierarchies.
* **Learning Focus**: Implement capsule networks and understand their structure.
* **Paper**: [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

### 🧠 **HRNet (2019)**

* **Key Innovation**: Maintained high-resolution representations through the network.
* **Learning Focus**: Implement high-resolution networks for tasks like segmentation.
* **Paper**: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919)

### 🧠 **NASNet (2018)**

* **Key Innovation**: Applied neural architecture search to design networks.
* **Learning Focus**: Implement neural architecture search techniques.
* **Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/1909.11059)

---

## Implementation Path

1. **Start with LeNet-5**: Understand the basics of CNNs by implementing a simple architecture.
2. **Progress to AlexNet and ZFNet**: Learn about deeper architectures and the importance of visualization.
3. **Implement VGGNet**: Understand the impact of depth and uniform architecture.
4. **Explore GoogLeNet and ResNet**: Learn about inception modules and residual connections.
5. **Dive into DenseNet**: Understand the benefits of densely connected layers.
6. **Work on MobileNet v1 and v2**: Learn about efficient architectures for mobile devices.
7. **Implement EfficientNet**: Understand model scaling and efficiency.
8. **Explore ConvNeXt**: Learn about modern architectural tweaks in CNNs.
9. **Optionally, experiment with Capsule Networks and HRNet**: Explore specialized architectures for specific tasks.

