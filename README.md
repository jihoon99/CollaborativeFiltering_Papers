# This Repository made for recap(get idea, review etc) State Of The Art papers(models)

# Read List
-[Blurring-Sharpening Process Models for Collaborative Filtering](https://arxiv.org/pdf/2211.09324v2.pdf)

# Done
## Collaobrative Filtering
- NGCF : [Neural Graph Collaborative Filtering](https://jihoonjung.tistory.com/63) -> [Explained](https://jihoonjung.tistory.com/63)
```
1. Implement Graph into Collaborative Filtering
2. This Paper studied modeling high-order information connections using graph theory.
```
- GraphSage : [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

- LightGCN : (2020, SIGIR) [Simplifying and Powering Graph ConvolutionNetwork for Recommendation](hhttps://arxiv.org/abs/2002.02126)
```
1. Graph has shown dramatic performance improvements on collaborative filtering problems, but it's not clear why (only experimental evidence)
-> We will chew, bite, and taste why Graph theory works so well in this study. (Activation Function, Feature Transformation perspective)
2. This Paper Propose a new architecture, LightGCN, based on the experimental results.
```
- CSE : [Collaborative Similarity Embedding for Recommender System](https://arxiv.org/pdf/1902.06188.pdf)
- UltraGCN : [Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2110.15114) -> [Explained](https://jihoonjung.tistory.com/65)
```
UltraGCN skips regressive message passing and uses a loss function to approximate the value after passing through an infinite number of message passing layers. This results in a convergence speed that is 10 times faster than previous studies and is ready for industrial application.
```
- SimpleX : [A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/pdf/1902.06188.pdf) -> [Explained](https://jihoonjung.tistory.com/65)
```
1. CF is composed of (1) interaction encoder, (2) loss function, and (3) negative sampling, but (2) and (3) have not been studied in detail in this research.
2. Proposed Cosine Contrastive Loss as a loss function
3. Designed a simple architecture called SimpleX
-> Performance improvement of up to 48.5% in NDCG@20 compared to LightGCN
```

- BSPM : [Blurring-Sharpening Process Models for Collaborative Filtering](https://arxiv.org/abs/2211.09324)
```
1. No trainable weights -> mathmetical functions are used
2. Bit All SOTA models.
```

## Collaborative Filtering With Training Embeddings
- RecVAE : [a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback](https://arxiv.org/abs/1912.11160) -> [Explained]()
- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)


## Prerequisite
- Denoising Diffusion Probabilistic Models : [DDPM](https://arxiv.org/abs/2006.11239)
- Score-Based Generative Modeling through Stochastic Differential Equations : [SDEs](https://arxiv.org/abs/2011.13456)
```
ref
- https://junia3.github.io/blog/scoresde
```
- DDIM
```
Through DDPM, it was possible to learn a generative Markov Chain Process in the form of generating samples by cutting through the noise. However, this process requires a lot of steps and is significantly slower than GAN, which ends in one step. Therefore, in this paper, we propose DDIM, which generalizes the process to non-markovian.
```
- Latent Diffusion Model(Stable Diffusion)
```
Must read VQGAN first
AutoEncoder is trained Based On VQGAN (w/ perceptual loss)
```