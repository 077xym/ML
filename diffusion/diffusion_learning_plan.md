# 🌀 Learning Diffusion Models: A Step-by-Step Guide

This guide outlines a structured plan to understand and implement diffusion models, especially from a deep learning practitioner's perspective.

---

## 🎓 Step 1: Intuition and Basics

| Resource | Type | What You'll Learn |
|---------|------|-------------------|
| [Lil'Log — Diffusion Models](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html) | Blog | Best introduction to diffusion models with visuals and math |
| [DDPM Original Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) | Paper | Introduces forward/reverse processes |
| [What are Diffusion Models? – AssemblyAI](https://www.youtube.com/watch?v=HoKDTa5jHvg) | Video | Beginner-friendly explanation with visuals |

---

## ⚙️ Step 2: Implementation from Scratch (PyTorch)

| Resource | Type | What You'll Learn |
|----------|------|-------------------|
| [Hugging Face Diffusers Course](https://huggingface.co/learn/diffusers-course/) | Free course | Hands-on course for training/using diffusion models |
| [DDPM in PyTorch – Phil Wang](https://github.com/lucidrains/denoising-diffusion-pytorch) | Code | Minimal and modular PyTorch implementation |
| [The Annotated Diffusion Model (Hugging Face)](https://huggingface.co/blog/annotated-diffusion) | Blog/code | Annotated PyTorch implementation |
| [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers) | Library | Industry-grade tools for diffusion workflows |

---

## 📐 Step 3: Advanced Topics & Variants

| Topic | Resource | Notes |
|-------|----------|-------|
| **DDIM (faster inference)** | [DDIM Paper](https://arxiv.org/abs/2010.02502) | Deterministic sampling |
| **Score-based models (SDEs)** | [Song et al., 2021](https://arxiv.org/abs/2011.13456) | General framework via SDEs |
| **Latent Diffusion** | [LDM Paper (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) | Compress to latent space |
| **Guidance (CFG, CLIP)** | [Classifier-Free Guidance Explained](https://k-d-w.org/blog/2022/11/07/classifier-free-guidance/) | Improves conditional generation |

---

## 💻 Step 4: Build Your Own Diffusion Pipeline

1. Train a DDPM on MNIST or CIFAR-10.
2. Add DDIM sampling and compare with vanilla DDPM.
3. Use `DiffusionPipeline` from HuggingFace to generate text-to-image.
4. Fine-tune Stable Diffusion on your own dataset using LoRA.

---

## 🧑‍🏫 Bonus: Tutorial Videos

- [Andrej Karpathy’s Diffusion Walkthrough](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [Aman’s Diffusion Deep Dive](https://www.youtube.com/watch?v=ZrKJnUbPf7w)

---