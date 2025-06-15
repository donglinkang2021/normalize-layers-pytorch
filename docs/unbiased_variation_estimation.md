# Unbiased Variation Estimation

Suppose we have a random variable $X$, whose expected value $\mu$ is defined as:

$$
\mu = \mathbb{E}[X]
$$

Now, given a set of samples $\{x\}$, we can estimate $\mu$ using the sample mean $\bar{X}$:

$$
\mu \approx \bar{X} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

Note that this is an approximation, meaning this estimate is an approximate value.

Now, if we want to estimate the variance $\sigma^2$ of $\mu$:

We have the definition of variance:

$$
\sigma^2 = \mathbb{E}[(X - \mu)^2]
$$

Based on the relationship between variance and expectation, we might naturally think that the relationship between sample variance and sample mean is similar, thus having:

$$
S^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{X})^2
$$

However, this sample variance $S^2$ is actually a biased estimate, as proven below:

$$
\begin{align*}
\mathbb{E}[S^2] &= \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{X})^2\right] \\
&= \mathbb{E}\left[
    \frac{1}{n} \sum_{i=1}^{n} \left(
        (x_i - \mu) - (\bar{X} - \mu)
    \right)^2
\right] \\
&= \mathbb{E}\left[
    \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\right] - 2 \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)(\bar{X} - \mu)\right] + \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^{n} (\bar{X} - \mu)^2\right] \\
&= \mathbb{E}\left[
    \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\right] - \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^{n} (\bar{X} - \mu)^2\right] \\
&= \textrm{Var}(X) - \textrm{Var}(\bar{X}) = \sigma^2 - \frac{\sigma^2}{n} = \frac{n-1}{n} \sigma^2\\
\end{align*}
$$

At the same time, we also obtain the unbiased estimate of $\sigma^2$:

$$
\begin{aligned}
S^2 &= \frac{n}{n-1} \cdot \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{X})^2 \\
&= \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2
\end{aligned}
$$

## TL;DR

1. Pay attention to the symbols:
   - $\mu$ is the expected value, $\sigma^2$ is the variance.
   - $\bar{X}$ is the sample mean, $S^2$ is the sample variance, which is an estimate.
2. The unbiased estimates for the sample mean and sample variance are as follows:

    $$
    \boxed{
        \begin{aligned}
        \mu &\approx \bar{X} = \frac{1}{n} \sum_{i=1}^{n} x_i \\
        \sigma^2 &\approx S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2
        \end{aligned}
    }
    $$