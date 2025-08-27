# Model Selection

Loss Function: $\alpha \times \mathcal{L}_\text{Focal-Tversky} + \mathcal{L}_\text{Weighted Cross-Entropy}$,

$$
\begin{split}
\mathcal{L}_\text{Focal-Tversky} &= \left( 1 - \frac{\mathbf{p}\cdot\mathbf{b}}{\mathbf{p}\cdot\mathbf{b} + \beta \bar{\mathbf{p}} \cdot \mathbf{b} + (1 - \beta) \mathbf{p} \cdot \bar{\mathbf{b}} \right)^\gamma\\
\mathcal{L}_\text{Cross-Entropy} &= -w_n \log p_n
\end{split}
$$

where $\mathbf{p}$ are the probabilities (softmaxxed outputs along channel dimension) from the model, and $\mathbf{b}$ are the binary mask (0, 1) values. We use $\bar{\cdot}$ to represent the complement, i.e. $\bar{x_i} = 1 - x_i$ in index notation.