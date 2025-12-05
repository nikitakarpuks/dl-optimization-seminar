Important
Better to write numpy implementation

Things to try:
1. ? relation with matrix factorization
2. ? precise forward and backward equations (possibly for a written report)
3. Efficient and inefficient gradient computation
4. Visualizations of GD convergence, plateaus, saddle points
5. Lipschitz smooth gradient, convergence dependent on beta (Lipschitz constant) [Proposition 1.2.3]
6. Visualization of vanishing/exploding gradients with simple examples; gradient clipping
7. ? Showing issue with RNNs
8. Initialization, variance distributions with different initialization techniques
9. Empirical variance and networks width/depth
10. Orthogonal initialization, dynamical isometry; resnet di
11. local Lipschitz constant of the gradient (the maximum eigenvalue of the Hessian)
12. Batch normalization, layer normalization; fixup initialization instead of BN for resnets
13. Choose of initial points for optimization
14. SGD parametrization and variants, their effect on optimization, convergence
15. simple quadratic problem instances that momentum does not improve the convergence speed of SGD
16. Non-convex problems
17. Adams insensitivity to hyperparameters
18. RMSProp and Adam can be divergent for convex problems
19. Training speed with multiple machines
20. Competition between second order methods and first-order methods. Adaptive gradient methods actually use second-order information implicitly
21. neural-net optimization is far from a worst-case non-convex problem, and finding a global minimum is not a surprise in deep learning noways
22. Lottery ticket hypothesis (LTH) states that such a winning ticket always exists
23. Wide/sharp minimas
24. Dep NN conditions for "every local-min is a global-min"
25. over-parameterized networks are prone to over-fitting
26. Neural Tangent Kernal
27. Finite-width Ultra-wide networks
28. Empirical computation by NTK
29. Mean-field approximation
30. Research in Shallow Networks after 2012


Seminar Project on Neural Network Training Dynamics – Plan and README
Overview

We will experimentally investigate key aspects of neural network training dynamics – especially initialization, variance propagation, and gradient flow – to complement the theoretical insights of the seminar paper. In particular, we focus on topics amenable to practical experiments and visualization, avoiding overly abstract purely dense-layer proofs. Our experiments will validate how initialization and layer‐wise variance affect learning stability, explore dynamical isometry via orthogonal initialization, compare RNN training with and without gating, and analyze the network Lipschitz constant via spectral norms. These topics connect directly to the paper’s themes (initialization schemes, stable signal propagation, gradient behavior)


. We will use PyTorch for most network experiments (for auto‐diff and GPU support) and NumPy for lightweight linear-algebra checks where appropriate.

Selected Topics (with Rationale)

Initialization & Variance Propagation (MLP) – How does the scale of random weights affect forward/backward signal? We will sweep weight variances in a ReLU MLP and compare Xavier vs. He/Kaiming schemes. This follows the paper’s E1/E2 narrative


 and illustrates that “inappropriate scales induce vanishing or exploding gradients, while a broad band around (ideal) yields stable learning”

. (Visualization: training curves and accuracy vs. init‐scale.)

Dynamical Isometry (Deep Networks) – Investigate whether using orthogonal initialization achieves near-isometry (Jacobian condition ≈1) and improves training of very deep nets. According to theory, orthonormal init makes the input-output Jacobian well-conditioned, enabling deep training (e.g. 10,000 layers!)
proceedings.mlr.press
. We will code a deep MLP/CNN and measure singular-value spectra of its Jacobian (before/after training) for orthogonal vs. random init, and compare gradient norms and convergence. (Expect orthogonal init to keep singular values near 1 and speed up convergence

proceedings.mlr.press
.)

Recurrent Nets – Gating vs. Vanilla RNNs – Examine trainability of RNNs with/without gating. The paper suggests vanilla RNNs suffer vanishing/exploding gradients except at a delicate “edge of chaos” (precise orthonormal init), whereas gated RNNs (e.g. LSTM/GRU or minimalRNN) have a much larger stable initialization region

. We will train a simple sequence task with (a) vanilla RNN and (b) gated RNN (e.g. GRU), using both random and orthogonal weight initializations. We will track training loss and hidden‐state gradient norms. (We expect the gated RNN to train reliably under wider conditions, and the vanilla RNN to fail unless carefully initialized

.)

Lipschitz Constants (Spectral Norms) – Measure how the global Lipschitz constant (a measure of sensitivity to input perturbations) evolves during training. The Lipschitz constant of a network is bounded by the product of its layers’ spectral norms (largest singular values). We will compute per-layer spectral norms (using PyTorch SVD) of trained MLPs under different regimes (e.g. with/without spectral normalization or weight decay) and plot how this estimate correlates with generalization or double-descent behavior. Recent work shows non-monotonic trends in empirical Lipschitz and links to test error

. (Visualization: Lipschitz estimate vs. epoch/test accuracy.)

These four topics provide a coherent narrative: from basic initialization/variance theory, through ensuring isometry and stable deep training, to specialized architectures (RNNs), and finally to functional sensitivity. Each will include code, empirical tests, and visualizations connecting back to the theory


.

Experimental Plans (per topic)
1. Initialization & Variance Propagation (MLP experiments)

What to code: A script (in PyTorch) to train a multi-layer perceptron (MLP) on a standard dataset (e.g. MNIST or CIFAR-10) with ReLU activations. We will implement a loop over initialization scales: initialize all weights from $\mathcal{N}(0,\sigma^2)$ with $\sigma$ swept logarithmically (e.g. $10^{-3}$ to $10^1$). We will also implement Xavier-normal and Kaiming-normal initializations.

What to explore: For each initialization, train for a fixed small number of epochs and record training loss and accuracy trajectories. Also compare runs with Xavier vs. He/Kaiming (fan-in) initialization under identical settings (architecture, optimizer, seed) to isolate initialization effects.

What to visualize: Plot final accuracy (and/or loss) vs. $\sigma$ to identify “vanishing” (low $\sigma$) and “exploding” (high $\sigma$) regimes, and the stable middle band

. Overlay representative loss/accuracy curves for selected $\sigma$ values. Additionally, plot the training loss curves for Xavier vs. Kaiming initializations on the same axes.

Expected results: Very small $\sigma$ should yield near-zero updates (vanishing signals), and very large $\sigma$ cause unstable training – matching [29], we expect a clear “sweet spot” of $\sigma$ where training is stable

. We anticipate Kaiming init will converge faster and more stably than Xavier for ReLU networks

 (as the theory predicts that fan-in initialization preserves forward variance, easing gradient flow


). We will cite the paper’s findings that “inappropriate scales induce vanishing or exploding gradients” and Kaiming outperforms Xavier under ReLU


.

2. Dynamical Isometry (Deep Feed-Forward Nets)

What to code: Build a very deep neural network (e.g. 50–100 layer MLP or CNN) in PyTorch. Implement two initialization modes: (a) standard Gaussian (He/Kaiming) and (b) orthogonal initialization (weights set to random orthonormal matrices, e.g. via PyTorch’s torch.nn.init.orthogonal_).

What to explore: Compute the input-output Jacobian $J = \frac{\partial f(x)}{\partial x}$ at initialization for a random input $x$ (or use a linearized approximation). Use NumPy or PyTorch SVD to find the singular values of $J$. Also monitor gradient norms of intermediate layers during the first few training steps. Then train both networks on the same task and compare convergence speed.

What to visualize: Bar chart or histogram of singular values of the Jacobian for each initialization (Gaussian vs. orthogonal). Plot gradient norm vs. layer index to see if gradients explode/vanish. Plot training loss curves for the two inits.

Expected results: Orthogonal initialization should yield singular values clustered near 1 (dynamical isometry) and thus keep gradients well‐conditioned
proceedings.mlr.press

. We expect the orthogonal‐init network to train faster and more robustly on very deep architectures. These experiments test the claim that “equilibration of singular values of the input-output Jacobian” (dynamical isometry) allows training of extremely deep networks
proceedings.mlr.press
. Indeed, networks with well-conditioned Jacobians have been shown to train orders of magnitude faster

. We will demonstrate this by comparing convergence and gradient flow between the two inits.

3. RNN Training Dynamics (Vanilla vs Gated)

What to code: Use PyTorch to implement or leverage built-in RNNs. We will train (a) a vanilla RNN and (b) a gated RNN (e.g. a GRU or a custom “minimalRNN”) on a sequence task (such as character-level language modeling on a small text or a synthetic sequence copying task). Both models will use identical hidden sizes and architectures aside from gating. Initialize weights either with Gaussian or with orthogonal matrices.

What to explore: Compare training behavior for vanilla vs. gated RNNs. Track hidden‐state gradient norms over time-steps or training epochs. Test sensitivity to initialization: use random Gaussian vs. orthogonal initial weight matrices.

What to visualize: Training loss curves for vanilla and gated RNNs (possibly on log scale). Plot of gradient norm over time or layer for both models. Optionally, accuracy or perplexity curves if applicable.

Expected results: We anticipate, in line with theory

, that the vanilla RNN will suffer vanishing/exploding gradients under random init and thus train poorly, whereas the gated RNN will train reliably. Using orthogonal init should significantly improve the vanilla RNN (pushing it toward the “edge of chaos”), but gating provides a much larger stable regime

. Thus, gated RNNs will typically converge faster and to lower loss. This matches the paper’s insight that gating yields a “robust subspace of good initializations” while vanilla RNNs only hit isometry on a thin manifold

. We will verify this by showing the vanilla RNN fails unless carefully initialized, while the GRU/LSTM learns stably.

4. Lipschitz Constant (Spectral Norm Analysis)

What to code: After training a feed-forward network (e.g. a small CNN or MLP), we will compute each layer’s spectral norm (largest singular value of the weight matrix). In PyTorch, we can use torch.linalg.svd or power iteration. We will do this for models trained under different conditions (e.g. standard training vs. training with spectral‐norm regularization).

What to explore: Investigate how the network’s Lipschitz bound (the product of per-layer spectral norms) evolves with training and relates to generalization. We may experiment with over-parameterized settings (varying width) to see double-descent effects.

What to visualize: Line plots of the (log) Lipschitz estimate vs. training epoch, and its correlation with test error or loss. Compare models with/without spectral normalization. Possibly plot double-descent curves annotated with corresponding Lipschitz values.

Expected results: We expect to observe that the empirical Lipschitz constant can exhibit non-trivial trends during training, often peaking before a descent in test error

. Prior work reports that higher Lipschitz (more sensitive models) often correlate with worse generalization

. By constraining spectral norms (or by using weight decay), the Lipschitz bound should shrink, potentially yielding smoother gradients and better generalization. This experiment will highlight how initialization and training dynamics affect network sensitivity, complementing the paper’s focus on keeping networks in well-conditioned regimes


.

Implementation Details

We will primarily use PyTorch for building and training networks because it simplifies gradient computation and offers easy initialization routines (e.g. torch.nn.init.kaiming_normal_, torch.nn.init.orthogonal_). NumPy will be used for smaller-scale linear algebra tasks (e.g. explicitly computing and analyzing Jacobians or spectral norms) when convenient. Each topic’s code can be organized into separate scripts or Jupyter notebooks. For example:

init_variance/ – contains code to sweep initialization scales and plot training metrics.

isometry_experiments/ – contains code for deep net with orthogonal vs Gaussian init and singular-value analysis.

rnn_dynamics/ – contains RNN training experiments (vanilla vs gated).

lipschitz_analysis/ – contains code for computing spectral norms and correlating with performance.

We will log all results (loss/accuracy histories, weight singular values) to files for reproducibility. All experiments will use fixed random seeds where appropriate to ensure consistency. Where possible, vectorized NumPy or PyTorch operations will be used for efficiency, but clarity of demonstration is the priority.

Project Structure (README Format)

The README for this project will outline:

Project summary: The goal of connecting theory (the seminar paper) with experiments in training dynamics.

Selected topics: As above, each topic is briefly described with its motivation (citing the paper when relevant).

Implementation structure: Folder organization and choice of tools (e.g. “Each topic has its own script; training networks in PyTorch, analysis in PyTorch/NumPy”).

Experiments: For each topic, a summary of the experiment steps (what is varied, what is measured) and the planned visualizations.

Expected outcomes: Key expected findings, such as stable variance regions and orthogonal init benefits for deep nets

proceedings.mlr.press
.

Usage: Instructions for running the code (not detailed here, but notionally in README).

This README will serve as the blueprint for the report and presentation, with references to the seminar paper’s results


.

Expected Results and Connections

We expect our experiments to yield non-trivial, illustrative outcomes aligned with theory. For instance, sweeping initialization scales should reveal a “vanishing-to-exploding” transition zone

; orthogonal initialization should enable very deep networks to train successfully
proceedings.mlr.press
; gating in RNNs should show a markedly larger stable regime for learning

; and the Lipschitz analysis should illustrate how network sensitivity evolves with training and model size

. Together, these results will reinforce the paper’s central story: that careful initialization (variance balancing, isometry) and architecture choices (gating, normalization) are crucial for stable training


.

References: Key insights guiding this plan include the seminar paper’s conclusions (e.g. “vanishing/exploding gradients when initialization is off, broad stable band exists”

, and “He/Kaiming outperforms Xavier under ReLU”

), studies on dynamical isometry (orthogonal init enabling 10k-layer nets
proceedings.mlr.press
), and RNN analyses showing gated units greatly improve trainability

. A recent study on Lipschitz constants highlights their correlation with generalization behavior

. These sources will be cited throughout our report and README.
