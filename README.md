# PromptViT-Percepta  
**Team Percepta’s project on prompt-tuned ViTs — bridging explainability and precision in fine-grained vision.**  

---

## 🧠 Project Overview  
This project explores **Prompt-Tuned Vision Transformers (ViTs)** for **Explainable Fine-Grained Recognition**.  
The goal is to improve classification accuracy on visually similar object categories (e.g., bird species, car models, flower types) while maintaining interpretability through **Prompt-CAM** and attention-based visual explanations.  

Our research focuses on balancing **performance** and **explainability**, demonstrating that lightweight prompt-tuning can provide interpretability without sacrificing precision.  

---

## 👥 Team Members  
| Name | Student ID | Email | Role |
|------|-------------|--------|------|
| Mirkomil Mirzohidov | 221408 | 221408@centralasian.uz | Model architecture & repo management |
| Muhammad Saidahmetov | 220838 | 220838@centralasian.uz | Experiments & evaluation metrics |
| Asilbek Tashpulatov | 221443 | 221443@centralasian.uz | Dataset preparation & report writing |

---

## 🎯 Objectives  
- Implement a **prompt-tuned Vision Transformer (ViT)** for fine-grained visual classification.  
- Generate **explainable visualizations** using attention heatmaps (Prompt-CAM).  
- Compare results with baseline fine-tuning and standard ViT models.  
- Evaluate trade-offs between **accuracy** and **interpretability**.

---

## 📚 Dataset  

We use publicly available fine-grained visual classification datasets:  

- [CUB-200-2011 (Birds)](https://www.kaggle.com/datasets/wenewone/cub2002011) — 200 bird species with subtle inter-class variations.  
- [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) — 16,185 images across 196 car categories.  
- [Oxford Flowers-102](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) — 102 flower categories with high intra-class similarity.  

These datasets are ideal for evaluating fine-grained classification performance and model explainability.

---

## ⚙️ Methodology  
1. **Baseline:** Fine-tune a pretrained Vision Transformer (ViT-B/16) on target datasets.  
2. **Prompt-Tuning:** Add learnable prompt tokens to the input embeddings and freeze most ViT parameters.  
3. **Explainability:** Use **Prompt-CAM** and attention rollout to visualize model focus regions.  
4. **Evaluation Metrics:** Accuracy, F1-score, and **Pointing Game** metric for localization quality.  

---

## 🧪 Experiments & Evaluation  
| Experiment | Description | Metric |
|-------------|--------------|--------|
| Baseline ViT | Standard fine-tuning on CUB-200 | Accuracy, F1 |
| Prompt-Tuned ViT | Lightweight tuning via prompts | Accuracy, F1 |
| Explainability | Evaluate interpretability (Prompt-CAM) | Pointing Game |
| Ablations | Compare # of prompts, layers tuned | Accuracy delta |

---

## 🧭 Roadmap  

| Week | Milestone | Owner | Due Date |
|------|------------|--------|----------|
| Week 1 | Team formation & topic selection | All | Oct 21 |
| Week 2 | Related work summary + dataset setup | Asilbek | Oct 27 |
| Week 3 | Baseline ViT training on CUB-200 | Mirkomil | Nov 3 |
| Week 4 | Prompt-tuning implementation | Muhammad | Nov 10 |
| Week 5 | Explainability (Prompt-CAM) integration | Mirkomil | Nov 17 |
| Week 6 | Evaluation, report draft | All | Nov 24 |
| Week 7 | Final proposal submission & repo polishing | All | Dec 1 |

🗂️ **ROADMAP.md** file will include weekly progress updates and issue tracking.

---

## 🛠️ Tech Stack  
- Python 3.10  
- PyTorch  
- HuggingFace Transformers  
- Matplotlib / Seaborn for visualization  
- Google Colab / Kaggle GPU runtime  
- **Visual Studio Code (VSCode)** — main development environment  

---

## ⚖️ Ethics & Compliance  
- All datasets used are **publicly available** and for research purposes only.  
- No personal or medical data will be collected.  
- Results and visualizations will be open-sourced for educational reproducibility.

---

## 📈 Expected Outcomes  
- Trained **Prompt-Tuned ViT** model for fine-grained classification.  
- Visual explainability results via **Prompt-CAM**.  
- Comparative report vs baseline fine-tuning.  
- Final deliverables: Code, trained weights, and presentation slides.

---

## 🧩 References  

- Chefer H., Gur S., Wolf L. *Transformer interpretability beyond attention visualization.*  
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 782–791.  

- Dosovitskiy A. *An image is worth 16x16 words: Transformers for image recognition at scale.*  
  arXiv preprint arXiv:2010.11929, 2020.  

- Jia M. et al. *Visual prompt tuning.*  
  In *European Conference on Computer Vision (ECCV)*. Cham: Springer Nature Switzerland, 2022, pp. 709–727.  

- Zhou B. et al. *Learning deep features for discriminative localization.*  
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 2921–2929.  


---

## 📜 License  
This project is conducted as part of the **Central Asian University — Computer Vision (Fall 2025)** course under academic fair use for research and educational purposes.

---

## 🌐 Repository Link  
[https://github.com/mirkomil06/PromptViT-Percepta](https://github.com/mirkomil06/PromptViT-Percepta)

