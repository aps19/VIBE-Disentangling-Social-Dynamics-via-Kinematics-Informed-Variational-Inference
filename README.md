<div align="center">

# 🎭 VIBE: Disentangling Social Dynamics via Kinematics-Informed Variational Inference for Behavioral Emotion

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue?style=for-the-badge&logo=academia&logoColor=white)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red?style=for-the-badge)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

**VIBE** is a kinematics-aware framework designed for Group Emotion Recognition (GER). Unlike current AI models that act as black boxes and get distracted by background environments, VIBE utilizes causal structuring and mathematical constraints to filter out background noise and isolate the genuine emotions and sociological mechanics of a crowd.

</div>

---

## 📰 News

- 🎉 **[May 2026]** VIBE has been **accepted at ICML 2026**!!
<!-- - 📄 **[May 2026]** Paper preprint is available on arXiv.
- 🚀 **[May 2026]** Code and pretrained models are released.  -->

---

## 🏗️ Architecture

VIBE is a multimodal, causally-structured framework that processes video, audio, and text through specialized encoders. The architecture is illustrated below:

![VIBE Architecture](VIBE_Architecture8.jpg)

*Overview of the proposed VIBE architecture. The framework processes features from three pre-trained encoders: Global Context (X_glob) via VideoMAE V2, Local Agent Tubes (X_loc) via DINOv2, and Acoustic Prosody (X_aud) via HuBERT. γ denotes the Synchrony Matrix used for dynamic gating, while X_text represents the textual anchor features. The Variational Information Bottleneck (VIB) compresses the representation and filters noise, while L_ortho enforces orthogonality between the environmental (Z_env) and affective (Z_aff) latent spaces to ensure disentanglement.*

---

## ✨ Key Innovations

- 🔀 **Causal Disentanglement**: Employs a Dual-Stream Variational Information Bottleneck (VIB) to strictly separate affective signals from environmental confounders.

- 📐 **Orthogonality Constraint**: Introduces a soft constraint that geometrically forces the "Agent" and "Context" latent spaces to remain disjoint, ensuring the model focuses on agent dynamics rather than background noise.

- ⚡ **Gamma-Gated Transformer**: Features a Transformer equipped with Decoupled Adaptive Layer Normalization (AdaLN). This modulates neural attention based on a raw physical synchrony metric ($\gamma$), allowing the model to dynamically adapt based on the crowd's physical state.

- 🔤 **Text-Guided Semantic Alignment**: Projects learned features into a shared embedding space to enforce cosine similarity with pre-trained textual anchors, explicitly grounding abstract visual dynamics in human interpretability.

---

## 🎞️ Architecture & Modalities

VIBE integrates audio, video, and text modalities through specialized pre-trained backbones:

| Modality | Feature | Backbone | Description |
|----------|---------|----------|-------------|
| 🎥 Video (Global) | $X_{glob}$ | VideoMAE V2 | Scene-level dynamics via token sequences |
| 👤 Video (Local) | $X_{loc}$ | DINOv2 | Agent-centric visual tubes per tracked person |
| 🔊 Audio | $X_{aud}$ | HuBERT | Paralinguistic cues (pitch, energy) |
| 📝 Text | $X_{text}$ | RoBERTa | Crowd-level semantic descriptions |

---

## 📊 Performance Benchmarks

VIBE sets a new state-of-the-art for group affect prediction on "in-the-wild" datasets, consistently outperforming prior methods:

| 📁 Dataset | 🎯 Accuracy | 📈 F1 Score |
|:---:|:---:|:---:|
| **VGAF** | **70.17%** | 0.701 |
| **GECV** | **91.85%** | 0.915 |

> 💡 Performance consistently peaks when the model tracks around **8 agents** ($K=8$), effectively capturing the primary affective circle.

---

## ⚙️ Implementation Setup

### 🖥️ Hardware Environment

- **GPU**: NVIDIA A2000 (12 GB VRAM)
- **CPU**: Intel® Xeon® W-2265
- **RAM**: 64 GB

### 🔧 Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Weight Decay | $1 \times 10^{-3}$ |
| LR Scheduler | Cosine Annealing |
| Batch Size | 16 |
| Initial LR | $1 \times 10^{-4}$ |
| Latent Dimension (D) | 512 |
| Epochs | 30 |
| Warm-up Period | 3 epochs |

### 📦 Data Processing

- 🎞️ Video streams are temporally decimated to **6 Hz**
- 📍 Raw trajectory coordinates are stabilized with a **Gaussian smoothing kernel** to mitigate detector jitter
- 🕒 Uniformly sampled at **T = 32 frames** per video clip
- 👥 Agent processing limit: **K_max = 8** agents per clip

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/VIBE.git
cd VIBE

# Install dependencies
pip install -r requirements.txt

# Run inference on a video
python inference.py --video path/to/video.mp4 --checkpoint checkpoints/vibe_best.pth
```

---

## 🧪 Reproduce Results

```bash
# Train on VGAF
python train.py --dataset vgaf --config configs/vgaf.yaml

# Train on GECV
python train.py --dataset gecv --config configs/gecv.yaml

# Evaluate
python eval.py --dataset vgaf --checkpoint checkpoints/vibe_best.pth
```

---

## 🌍 Impact

By effectively separating human behavior from the surrounding environment, VIBE mitigates **"shortcut learning"** — preventing models from making biased guesses based on background details. This helps build more **transparent**, **accountable**, and **responsible** AI systems for emotion recognition that are robust across diverse, real-world locations.

---

## 📜 Citation

If you find VIBE useful in your research, please cite our work:

```bibtex
@inproceedings{vibe2026,
  title     = {VIBE: Disentangling Social Dynamics via Kinematics-Informed
               Variational Inference for Behavioral Emotion},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built by Abhishek and Deepak  | Accepted at ICML 2026</sub>
</div>



<!--
## VIBE: Disentangling Social Dynamics via Kinematics-Informed Variational Inference for Behavioral Emotion 

**VIBE** is a kinematics-aware framework designed for Group Emotion Recognition (GER). Unlike current AI models that act as black boxes and get distracted by background environments , VIBE utilizes causal structuring and mathematical constraints to filter out background noise and isolate the genuine emotions and sociological mechanics of a crowd.

---

## Key Innovations

* **Causal Disentanglement**: Employs a Dual-Stream Variational Information Bottleneck (VIB) to strictly separate affective signals from environmental confounders.


* **Orthogonality Constraint**: Introduces a soft constraint that geometrically forces the "Agent" and "Context" latent spaces to remain disjoint, ensuring the model focuses on agent dynamics rather than background noise.


* **Gamma-Gated Transformer**: Features a Transformer equipped with Decoupled Adaptive Layer Normalization (AdaLN). This modulates neural attention based on a raw physical synchrony metric ($\gamma$), allowing the model to dynamically adapt based on the crowd's physical state.


* **Text-Guided Semantic Alignment**: Projects learned features into a shared embedding space to enforce cosine similarity with pre-trained textual anchors, explicitly grounding abstract visual dynamics in human interpretability.



---

## Architecture & Modalities

VIBE integrates audio, video, and text modalities through specialized pre-trained backbones:

* **Global Context ($X_{glob}$)**: Captured using VideoMAE V2 to process scene-level dynamics.


* **Local Agent Tubes ($X_{loc}$)**: Visual tubes are extracted for tracked agents and processed via DINOv2 to represent raw actor dynamics.


* **Acoustic Prosody ($X_{aud}$)**: Paralinguistic cues (like pitch and energy) are extracted using HuBERT.


* **Textual Context ($X_{text}$)**: Crowd-level descriptions are encoded using RoBERTa to serve as linguistic anchors for the semantic alignment objective.



---

## Performance Benchmarks

VIBE sets a new benchmark for group affect prediction, consistently outperforming state-of-the-art methods on "in-the-wild" datasets:

| Dataset | Accuracy | F1 Score |
| --- | --- | --- |
| **VGAF** | <br>**70.17%** | 0.701 |
| **GECV** | <br>**91.85%** | 0.915 |

Performance consistently peaks when the model tracks around 8 agents ($K=8$), effectively capturing the primary affective circle.

---

## Implementation Setup

* **Hardware Environment**: Conducted on a workstation equipped with an NVIDIA A2000 GPU (12GB VRAM), an Intel® Xeon® W-2265 CPU, and 64 GB of RAM.


* **Optimization**: Trained using the AdamW optimizer with a weight decay of $1\times10^{-3}$ and a Cosine Annealing learning rate scheduler.


* **Hyperparameters**: Utilizes a batch size of 16, an initial learning rate of $1\times10^{-4}$, and a latent dimension (D) of 512, trained over 30 epochs.


* **Data Processing**: Video streams are temporally decimated to 6 Hz, and raw trajectory coordinates are stabilized using a Gaussian smoothing kernel to mitigate detector jitter.



---

## Impact

By effectively separating human behavior from the surrounding environment, VIBE mitigates "shortcut learning," preventing models from making biased guesses based on background details. This helps build more transparent, accountable, and responsible AI systems for emotion recognition that are robust across diverse, real-world locations.
