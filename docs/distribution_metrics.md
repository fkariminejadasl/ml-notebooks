# Metrics for Comparing Distributions

This document summarizes common metrics used to compare probability distributions.  
These measures are often applied in statistics, machine learning, and information theory  
to quantify similarity, dissimilarity, or divergence between distributions.

---

## 1. Kullback–Leibler (KL) Divergence
- **Type:** Divergence (not symmetric, not a true distance).
- **Definition:**  
  `D_KL(P‖Q) = Σ P(x) log(P(x) / Q(x))`
- **Interpretation:** Measures how much information is lost when distribution **Q** is used to approximate **P**.
- **Notes:** Asymmetric; can be infinite if `Q(x) = 0` where `P(x) > 0`.

---

## 2. Jensen–Shannon Divergence (JSD)
- **Type:** Symmetrized and smoothed version of KL divergence.
- **Definition:**  
  `D_JS(P‖Q) = ½ D_KL(P‖M) + ½ D_KL(Q‖M)` where `M = ½(P + Q)`
- **Interpretation:** Bounded between 0 and 1 (when using log base 2).  
  Often used in clustering and GAN training.
- **Notes:** Square root of JSD is a proper metric.

---

## 3. Cross-Entropy
- **Type:** Expectation-based measure.
- **Definition:**  
  `H(P, Q) = - Σ P(x) log Q(x)`
- **Interpretation:** Expected number of bits to encode samples from **P** when using a code optimized for **Q**.
- **Notes:** Common in machine learning as a loss function (e.g., classification tasks).

---

## 4. Bhattacharyya Coefficient & Distance
- **Coefficient:**  
  `BC(P, Q) = Σ sqrt(P(x) * Q(x))`  
  (measures overlap between distributions; ranges from 0 to 1).
- **Distance:**  
  `D_B(P, Q) = -ln(BC(P, Q))`
- **Interpretation:** Higher overlap → smaller distance.  
  Used in pattern recognition and Bayesian classification.

---

## 5. Earth Mover’s Distance (EMD) / Wasserstein Distance
- **Type:** True metric (for certain conditions).
- **Definition:** Informally, the minimal "cost" of transforming one distribution into another,  
  where cost is the amount of probability mass moved times the distance it is moved.
- **Interpretation:** Reflects differences in support and geometry.  
  Often used in computer vision and GANs (Wasserstein GAN).
- **Notes:** Computationally more expensive than KL or JSD.

---

## 6. Hellinger Distance
- **Type:** True metric.
- **Definition:**  
  `H(P, Q) = (1/√2) * sqrt( Σ ( sqrt(P(x)) - sqrt(Q(x)) )² )`
- **Interpretation:** Related to Bhattacharyya coefficient.  
  Ranges between 0 (identical) and 1 (maximally different).
- **Notes:** Symmetric and bounded.

---

## 7. Total Variation (TV) Distance
- **Type:** Metric.
- **Definition:**  
  `D_TV(P, Q) = ½ Σ |P(x) - Q(x)|`
- **Interpretation:** Maximum difference in probabilities assigned by **P** and **Q** over all events.  
  Represents the largest discrepancy in outcome probabilities.
- **Notes:** Bounded between 0 and 1.

---

## 8. Maximum Mean Discrepancy (MMD)
- **Type:** Kernel-based metric (generalization of Total Variation).
- **Definition:**  
  `MMD(P, Q; k) = || E_P[k(x,·)] - E_Q[k(x,·)] ||_H`  
  where `k` is a kernel (e.g., Gaussian RBF), and `H` is the corresponding reproducing kernel Hilbert space (RKHS).
- **Interpretation:** Measures how distinguishable distributions **P** and **Q** are when mapped into a feature space defined by the kernel.  
  With a universal kernel, MMD = 0 if and only if P = Q.
- **Relation to TV:** If the kernel is chosen as a Dirac delta, MMD reduces to the Total Variation distance.
- **Notes:** Commonly used in two-sample tests, domain adaptation, and Generative Models (e.g., MMD-GAN).

---

| Metric                   | Symmetric | Bounded | True Metric | Notes |
|---------------------------|-----------|---------|-------------|-------|
| KL Divergence             | ✗         | ✗       | ✗           | Asymmetric, infinite possible |
| Jensen–Shannon Divergence | ✓         | ✓       | √(JSD) only | Smoothed KL, used in ML |
| Cross-Entropy             | ✗         | ✗       | ✗           | Common ML loss |
| Bhattacharyya Distance    | ✓         | ✓       | ✗           | Based on overlap |
| Earth Mover’s (Wasserstein)| ✓        | ✓       | ✓           | Captures geometry |
| Hellinger Distance        | ✓         | ✓       | ✓           | Related to Bhattacharyya |
| Total Variation           | ✓         | ✓       | ✓           | Intuitive probability gap |
| Maximum Mean Discrepancy  | ✓         | Depends | ✓           | Kernelized extension of TV |

---

## Further Reading and Notes

- In [*Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning*](https://arxiv.org/abs/2509.04734), the choice of distribution metric significantly impacts the quality of learned representations. 


