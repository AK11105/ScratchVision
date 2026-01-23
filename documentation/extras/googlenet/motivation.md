### Issue with standard scaling (increasing height and width)

Increase Depth ==> harder optimization, vanishing gradients

Increase Width ==> quadratic compute + parameter blow-up

Fully Connected Layers ==> dense interactions everywhere

**Problem is not that bigger models overfit**, it is **Dense connectivity assumes that everything depends on everything else.** which is **NOT TRUE**

### Sparse Connectivity

> Does not mean sparsity in weights (like L1 regularization).

Meaning is : **Topological Sparsity**
- Not all neurons talk to each other
- Connectivity reflects statistical dependence, not convenience

In case of images ==> nearby pixels are highly correlated, distant ones are weakly correlated

This matches: locality, compositonal structure, spatial coherence

### Why does Sparse Connectivty work?

Works because: If the probability distribution of the dataset is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statistics of activations of last layer and clustering neurons with highly correlated outputs.

- **representable?** : There exists some network — not necessarily learned yet — whose connectivity graph mirrors the dependency structure of the data. This is an assumption, We assume:
    - Image features form clusters of mutual dependence
    - Not all features interact directly
    - Interactions are localized and structured

- **Correlation statistics --> topology**: 
    - Suppose 2 neurons activate together consistently, That means they are responding to the same underlying image structure, Therefore, they should be grouped or connected earlier. ==> Clustering Arguement.
    - Analogy: Think of edges
        - Vertical edge detectors fire together
        - Corners fire when specific edges co-occur
        - Object parts fire when certain corners + textures co-occur
    - Dense connections are wasteful because:
        - Most features are conditionally independent
        - Only a few interactions matter per concept

### Why Dense layers are statistically wrong for images?

FC layer assumes: Every feature might matter for every other feature, equally.

But in images:
- A cat’s whisker does not directly depend on a car’s wheel
- Dependencies are hierarchical and localized

Dense layers ignore: spatial separation, scale separation, compositional structure

Thus:
- They waste parameters modeling nonexistent dependencies
- They increase overfitting risk
- They inflate computation quadratically

### Why sparsity reduces overfitting

Sparse connectivity:
- restricts hypothesis space
- enforces locality and modularity
- prevents accidental co-adaptation

This is inductive bias, not regularization in the loss.

We are forbidding the model from expressing implausible explanations.

### Relation to Hebbian principle

**Neurons that fire together, wire together**

What they mean:
- Statistical correlation implies shared cause
- Shared cause implies shared representation
- Shared representation implies closer connectivity

In modern words: Correlated activations suggest a latent factor that should be modeled jointly.

### How does this help Inception?

Given:
- Image features cluster by scale and type
- Dependencies are sparse
- Multiple scales coexist

Then:
- You want parallel sparse subgraphs
- Each subgraph specializes
- Their outputs are combined without forcing interaction everywhere

That is exactly what an Inception module is:
- sparse
- multi-branch
- scale-partitioned
- locally dense, globally sparse.


```
If images did not have sparse dependency structure: 
- Inception would fragment information
- Important cross-scale interactions would be delayed or lost

We would see:
- inconsistent activations
- brittle recognition
- dependence on late fusion

This becomes relevant later for global reasoning failures.
```