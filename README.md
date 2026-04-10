# R2A-Attack: Route-to-Rome Attack

**Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization**

> 🚧 **Code coming soon!** We are preparing the official implementation and will release it here shortly.

---

## Overview

**R2A-Attack** (Route-to-Rome Attack) is an adversarial attack targeting LLM routing systems. Modern deployments of large language models often use a **router** to direct incoming queries to either a cheaper (smaller) model or a more expensive (larger) model based on the estimated complexity of the query. R2A-Attack demonstrates that an adversary can craft adversarial suffixes and append them to queries so that the router is systematically misled into forwarding those queries to the expensive, high-capability model — even when the cheaper model would have been sufficient.

This work exposes a previously underexplored attack surface in LLM serving infrastructure and raises important security concerns for cost-sensitive LLM deployments.

---

## Key Contributions

- **Novel attack formulation**: We define the Route-to-Rome (R2A) threat model, in which an attacker manipulates routing decisions by optimizing adversarial suffixes appended to user queries.
- **Adversarial suffix optimization**: We propose a gradient-based optimization method to craft transferable suffixes that reliably redirect queries to expensive models across different router architectures.
- **Empirical evaluation**: We evaluate R2A-Attack against several representative LLM routing systems and demonstrate high attack success rates with minimal perturbation.
- **Security implications**: We discuss the economic and operational impact of routing attacks and outline potential defenses.

---

## Method

The core idea of R2A-Attack is to optimize a short adversarial suffix `s*` such that, when appended to a benign query `q`, the resulting string `q ⊕ s*` is routed to the expensive model by the target router. The optimization objective is:

```
s* = argmax_{s} P(route(q ⊕ s) = expensive_model)
```

We leverage gradient information from differentiable routing models to iteratively update the suffix tokens, making the attack both effective and efficient. The optimized suffixes are shown to transfer across different query types and router configurations.

---

## Attack Pipeline

```
User Query (q)
      │
      ▼
 Append Adversarial Suffix (s*)   ◄── Optimized via gradient-based search
      │
      ▼
 LLM Router  ──►  Expensive Model  (attack goal)
                  (instead of cheap model)
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{r2a_attack,
  title   = {Route-to-Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization},
  author  = {Authors},
  journal = {Venue},
  year    = {2025}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.

---

> 📬 For questions or collaborations, feel free to open an issue or reach out via GitHub.
