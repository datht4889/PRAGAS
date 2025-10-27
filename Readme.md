# PRAGAS: Enhancing Few-Shot Continual Relation Extraction via Pairwise Augmentation and Guided Sharpness-Aware Optimization
## Project Structure

```
mutual-pairing-data-augmentation/
├── FCRE/              # Few-Shot Continual Relation Extraction
│   ├── CPL/           # Contrastive Prompt Learning
│   └── SIRUS/         # Similar Relation Clusters
└── CED/               # Continual Event Detection
    └── SharpSeq/      # Sharp Sequence Learning
```

---

## Few-Shot Continual Relation Extraction (FCRE)

### Setup

1. **Change directory**
```bash
cd FCRE
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

### Running Experiments

#### CPL (Curriculum-based Prompt Learning)

**With BERT encoder:**
```bash
cd CPL
bash bash.sh
```

**With Large Language Models (LLM):**
```bash
cd CPL
bash bash_llm.sh
```

#### SIRUS Baseline

**With BERT encoder:**
```bash
cd SIRUS/BERT/bash
bash fewrel_5shot.sh    # For FewRel dataset
bash tacred_5shot.sh    # For TACRED dataset
```

**With Large Language Models (LLM):**
```bash
cd SIRUS/LLM/bash
bash fewrel_5shot.sh    # For FewRel dataset
bash tacred_5shot.sh    # For TACRED dataset
```

---


## Continual Event Detection (CED)

### Setup

1. **Change directory**
```bash
cd CED
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

### Running Experiments

**SharpSeq on MAVEN dataset:**
```bash
bash sh/maven.sh
```

---
## Citation

If you use this code in your research, please cite:

```bibtex
@article{pragas2024,
  title={PRAGAS: Enhancing Few-Shot Continual Relation Extraction via Pairwise Augmentation and Guided Sharpness-Aware Optimization},
  author={Hoang Thanh Dat, Nguyen Hoang Anh, Nam Le Hai, Linh Ngo Van, Sang Dinh},
  year={2024}
}
```