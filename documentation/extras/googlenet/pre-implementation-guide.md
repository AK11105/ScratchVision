# Pre Implementation Mental Model

## The Atomic Unit: Inception Block

### What Enters an Incpetion Block

A 3D feature tensor (H * W * C) representing spatially aligned set of feature activations, where H and W encode image location and C encodes a learned basis of feature primitives.

The module makes no assumptions about what those channels represent beyond locality and translation equivariance.

### What tensor leaves it

Output tensor is of shape (H * W * C') where H and W are identical to input, and C' is the sum of channels produced by each branch.

Spatial dimensions must remain unchanged so that features computed at different scales remain co-located in image space and can be jointly interpreted in subsequent layers.

### What does "parallel" mean

Parallel means each branch operates on the same input tensor, but computes its output independently, with no intermediate information exchange.

This forbids early cross-scale entanglement and encodes the assumption that visual structures at different spatial scales should be stabilized independently before being combined.

### What information allowed to mix

Information is allowed to mix only at the point of depth-wise concatenation, where features from different scales are co-located in the channel dimension but not yet combined.

This is a declarative form of cross-scale mixing: scales are made jointly available, but any interaction between them must be learned explicitly in subsequent layers.

## Branch Discipline

### During Forward pass, Can branches see other branch's output before concatenation?

**NO**

Visual structures at different spatial scales correspond to different latent causes and should be stabilized independently before any interaction.

In other words:

- A texture hypothesis should not adapt based on a contour hypothesis before it has settled
- A part-scale detector should not depend on object-scale context at the same stage

### If allowed to interact early, what would that imply

If branches interact before concatenation, you would observe:
- scale leakage
- loss of specialization
- redundant or unstable mid-level features
- delayed abstraction

Practically: the network degenerates into a wide, expensive convolution

**Inception loses its raison d’être**

## 1X1 Convolutions Placement

### Before 3X3

Yes

- It reduces channel dimensionality, not spatial resolution
- Assumes redundancy across channels
- Preserves spatial locality
- Makes expensive spatial aggregation tractable

What it preserves

- spatial alignment
- translation equivariance

What it compresses

- channel basis
- feature co-activity redundancy

### Before 5X5

Yes, Same reasoning as above

If not applied
- 5×5 convolutions explode compute
- The network avoids large receptive fields entirely
- Scale diversity collapses

### After Pooling

Yes

- Pooling destroys phase information
- Pooling emphasizes presence over alignment
- Pooling outputs are statistically different from conv outputs

Thus, 1x1 conv
- re-encodes pooled responses
- restores learned sparsity
- aligns pooled features with other branches

### After Concatenation

Yes

- Concatenation only juxtaposes features
- It does not model interactions

A subsequent 1x1 would allow

- combine scales selectively
- suppress irrelevant scales
- learn cross-scale info explicitly

Paper does not enforce this after concat in every module, but it is fundamentally allowed

### Inside Auxillary Features

Yes

- dimensionality reduction
- reparameterization
- non-linearity
- prevents auxiliary heads from overpowering main trunk

### Everywhere for Efficiency

No

If allowed
- the model becomes channel-mixing dominant
- spatial aggregation weakens
- the architecture drifts toward MLP-like behavior

## Spatial Alignment Contract

### Spatial Resolution

Must be preserved

All branches must emit feature maps that are pixel-aligned so that concatenation corresponds to the same image locations.

### Stride across branches

- All convolutional branches inside an Inception module use stride = 1
- Pooling inside an Inception module also uses stride = 1
- Downsampling is never implicit inside an Inception block
- Downsampling happens only between stages, via dedicated pooling layers

### Padding

Rule to follow for padding: $\frac{kernel \, size - 1}{2}$

Anything else causes spatial drift

- 1x1 P=0
- 3x3 P=1
- 5x5 P=2

### Pooling Behavior

Parallel Branch, Stride=Padding=1

- Pooling must not downsample
- Pooling introduces local invariance, not resolution change
- Resolution change is a stage-level decision, not a branch-level one

### What happens if even one branch violates

**Spatial phase misalignment leading to semantic noise**

What that looks like:
- features at the same spatial index refer to different image locations
- edges from one branch align with textures from another
- subsequent convolutions combine incoherent evidence

This produces:
- ghosting-like effects
- unstable mid-level features
- degraded localization
- brittle recognition

This is worse than blur or aliasing — it is semantic corruption.

## Pooling as a branch

### Information Loss

Pooling:
- destroys spatial phase
- collapses fine-grained geometry
- keeps “presence” not “arrangement”

### Optional Invariance

By placing pooling in parallel:
- invariance is offered, not imposed
- other branches preserve precise spatial detail
- the network can choose when invariance helps

### Architectural Honesty

Parallel pooling is an honest admission that:
- invariance is a hypothesis, not a fact
- some features benefit from it, others don’t
- forcing it globally would be dishonest modeling

### What happens if pooling forced first

The assumption violated is that early visual representations should preserve fine spatial structure until sufficient abstraction is achieved.

If pooling were forced first:
- all branches would inherit invariance
- fine-scale hypotheses would never form
- the model would assume “location doesn’t matter” too early

Visually:
- small objects vanish
- parts lose geometry
- textures collapse into blobs

## #3x3 Reduce and #5x5 Reduce

They are

- Empirical hyperparameters
    - Chosen experimentally
    - Tuned to balance representation and compute
    - Dataset- and depth-dependent

- Hardware compromises
    - Control FLOPs and memory
    - Make 5×5 convolutions feasible
    - Prevent channel explosion after concatenation

- Statistical assumptions
    - Assume channel redundancy
    - Assume that information can be compressed temporarily
    - Assume that important structure survives projection

### If reduction too small

If reduction is too small:
- channel basis collapses
- distinct visual cues are merged
- 3×3 and 5×5 filters receive impoverished signals

Visually:
- different textures look the same
- orientations blur together
- phase information is lost before spatial aggregation

This leads to:

**Under-diversified scale features**

### If reduction too large

If reduction is too large:
- almost no compression occurs
- expensive convolutions see high-dimensional input
- concatenation causes channel explosion

Visually:
- branches over-specialize
- noisy or spurious correlations are preserved
- later layers must disentangle unnecessary detail

This leads to:

**Feature clutter and inefficient abstraction**

## Auxilliary Classifiers

### Parameter Sharing with main trunk

Auxiliary classifiers branch off the main trunk and introduce new parameters that exist only during training.

If parameters were shared:
- the auxiliary objective would directly constrain the final classifier
- representation pressure would be inconsistent
- the model would no longer have a clean abstraction hierarchy

Instead:
- auxiliary heads shape representations
- but do not define them

### Gradient Recieved from final classifier

The final classifier backpropagates gradients through the entire trunk
- Each auxiliary classifier backpropagates gradients only through the layers before its attachment point
- There is no gradient flow from the final head into the auxiliary heads
- There is no gradient flow from auxiliary heads into later layers

This creates:
- shorter supervision paths
- localized gradient injection
- improved training stability

## Final Classifier

### Why Global Average Pooling --> Linear

GAP assumes presence > precise position
- It assumes spatial redundancy at high abstraction
- It assumes the network has already:
    - localized parts
    - aggregated context
    - resolved geometry

### Why reject FC

Fully connected layers assume:
- dense, unstructured dependency
- fixed spatial layout
- global mixing without locality

This violates:
- sparsity hypothesis
- translation equivariance
- scale modularity

### Why GAP --> Linear and not GAP --> MLP

An MLP after GAP would:
- reintroduce dense feature interactions
- allow arbitrary co-adaptation of semantic channels
- act as a hidden fully connected classifier

GoogLeNet’s philosophy is:
- the trunk learns structure
- the classifier only reads out evidence

### Why Conv --> FC also rejected

Even if conv preserves locality:
- flattening re-destroys spatial structure
- FC reintroduces fixed-size dependence
- classifier becomes geometry-sensitive again

This undoes the entire architectural effort.

### What assumption breaks if input resolution changes at inference?

If input resolution changes at inference, this assumption breaks:

**That the spatial statistics seen during training match those at inference.**

More concretely:
- receptive fields cover different fractions of the image
- object scale relative to feature maps changes
- “presence” becomes ambiguous

So while:
- FC mechanically breaks
- GAP semantically degrades

GAP tolerates resolution change better, but does not make the model scale-invariant.

