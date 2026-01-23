### Limitation of Sparse Datastructures

```markdown
Computing infrastructures are very inefficient when it comes to numerical calculation on non-uniform sparse data structures.

Even if number of arithmetic operations is reduced by 100x, the overhead for lookups and cache misses is so dominant that switching to sparse ==> **not worth it**

This gap widened further by **libraries tuned for dense matrix calculation** 
```

### Current vision oriented ML system utilize sparsity in spatial domain just by the virtue of employing convolutions.

This does not mean that **CNNs are sparse networks**

It means: **What sparsity exists conceptually**

A convolution layer enforces:

- Local connectivity: Each output unit depends only on a small spatial neighborhood (receptive field)

So compared to a fully connected layer:
- Most possible pixel–pixel interactions are forbidden
- Only nearby spatial interactions are allowed

**This is topological sparsity in the spatial domain.**

Visually:
- A neuron “sees” a patch, not the whole image
- This matches natural image statistics

So conceptually, CNNs already implement one kind of sparsity: **Spatial sparsity**

### Why does this *spatial sparsity* dissapear in practice

Convolutions are implemented as collections of dense connections to the patches in the earlier layer.

Inside a convolution operation:
- The receptive field is small (e.g., 3×3)
- But inside that patch, the computation is fully dense
- Every input channel connects to every output channel

So:
- Sparse globally
- Dense locally

And hardware does not see *local receptive fields* or *sparse graphs*, it sees **dense matrix multiplications and contigous memory blocks**


### Feature dimension sparsity vs Spatial sparsity

> “Traditionally ConvNets have used random and sparse connection tables in feature dimensions…”

This refers to channel-wise sparsity, not spatial.

Historically:
- Some early CNNs connected only subsets of channels
- This broke symmetry and reduced parameters

> “…but moved back to fully connected to utilize parallel compute.”

Why?

Because irregular sparsity is poison to hardware:
- cache misses
- pointer chasing
- non-coalesced memory access

Even if math ops ↓ 100×, memory stalls dominate.

So the industry choice was: Waste computation, save wall-clock time.

### Dilemma

Theory says: Vision structure is sparse and clustered.

Hardware says: Only dense, regular computation is efficient.

So the question becomes: Can we express a sparse vision hypothesis using dense computation primitives?

### Clustering sparse matrices into dense submatrices

Instead of: One huge sparse matrix (bad for hardware)

Use: 
- Multiple small dense blocks
- Each block corresponds to a cluster of correlated features

This preserves 
- sparsity at the global level
- density at the local (compute) level

### Why Inception

Inception is not random creativity and multi branch CNN

**It is a hardware-realizable approximation of a sparse vision graph.**

Each branch is:
- dense internally (GPU-friendly)
- specialized (scale / pooling / channel mixing)

Between branches:
- no interaction
- no forced coupling

> Reason for no interaction: If interactions allowed, output of different scale mixed before specialization, unrelated scaled would be forced to correlate, falls back to dense cross-scale dependencies ==> Doing this destroys the clustering that makes sparsity meaningful.

So globally:
- sparse connectivity graph

Locally:
- dense computation blocks

This is exactly the “clustered sparsity” idea.

### Summary

- Convolutions → spatial sparsity
- Inception → feature sparsity via clustered dense blocks
- Hardware → demands regular dense computation

**Inception exists at the intersection of all three**