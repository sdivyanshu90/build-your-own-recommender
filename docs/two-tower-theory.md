# Two-tower theory

Let train-fitted feature transforms produce user inputs $x_u$ and item inputs $x_i$. Independent
neural networks map them to a shared $d$-dimensional space:

$$z_u=f_\theta(x_u),\quad z_i=g_\phi(x_i)$$

For normalized embeddings, $u=z_u/\lVert z_u\rVert_2$ and
$v=z_i/\lVert z_i\rVert_2$. Inner product then equals cosine similarity:

$$s(u,i)=u^Tv=\cos(u,v)$$

Without normalization, dot product also depends on vector magnitude. That may encode confidence or
popularity, but can allow large norms to dominate and makes ANN thresholds less stable.

For a batch of $B$ observed positive pairs, the asymmetric in-batch probability is:

$$p(i_j\mid u_j)=\frac{\exp(s(u_j,i_j)/\tau)}
{\sum_{k=1}^{B}\exp(s(u_j,i_k)/\tau)}$$

and retrieval loss is:

$$\mathcal{L}=-\frac{1}{B}\sum_{j=1}^{B}\log p(i_j\mid u_j)+\lambda\lVert\Theta\rVert_2^2$$

Temperature $\tau$ controls how sharply the loss emphasizes close negatives. This implementation
uses a multi-positive numerator when an item repeats in a batch, preventing duplicate positives
from being labeled as negatives. A symmetric variant additionally predicts users from items; it
can improve bidirectional alignment but is not required for user-to-item serving.

The gradient pulls a user's representation toward positive item representations and pushes it
away from sampled alternatives. Repeated updates organize regions around features and behaviors
that help distinguish positives. The vectors do not contain an intrinsic semantic guarantee;
their meaning is defined by labels, sampling, architecture, and bias in exposure data.

Tower independence is the scaling advantage: every eligible item vector can be precomputed.
Online work is one user-tower pass plus ANN search, instead of running a neural network for every
user-item pair. Hard negatives improve fine discrimination but can include false negatives.
Collapse is monitored through norms and similarity distributions; low temperature, weak negative
diversity, or label defects can aggravate it.

Do not use a two-tower-only stack when exact cross-feature interactions dominate, the catalog is
small enough for exhaustive rich scoring, labels are too sparse or unstable, or strict policy
logic cannot be separated from learned relevance.
