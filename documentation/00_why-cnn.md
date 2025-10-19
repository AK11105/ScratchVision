# Rationale for CNN

## Why not use Multi-Layer-Perceptron (MLP)

### **Approach using MLP:** 

- Convert the 2D/3D image into a 1D vector ==> `nn.Flatten`
- Feed into the MLP i.e. vector becomes input to fully connected network

### **Why this is not optimal**

- **Massive number of parameters:** Eg. 100x100 RGB image ==> 100\*100\*3 = 30,000 input features and let's say first hidden layer has 1000 neurons ==> 30,000,000 parameters. This leads to
    - slow training
    - massive memory usage
    - high risk of overfitting

- **Destruction of spacial structure:** Flattening of images throws away all spatial information.
    - A pixel at top left is treated same as pixel as bottom right even though their spatial relationship is critical for understanding the image

- **Lack of Translational Invariance:** MLPs are not robust to object's position. It learns specific weights for features at specific locations.
    - Eg. If translated/rotated by few pixels, MLP trained on earlier img will not be be able to classify this updated image correctly

### CONVOLUTIONAL NEURAL NETWORK SOLVES THIS