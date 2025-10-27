# PRAGAS: Enhancing Few-Shot Continual Relation Extraction via Pairwise Augmentation and Guided Sharpness-Aware Optimization

## FCRE
### Change directory
```bash
cd FCRE
```

### Install requirements
```bash
pip install -r requirements.txt
```

### Run CPL (with BERT and LLM)
```bash
cd CPL
bash bash.sh
bash bash_llm.sh
```

### Run SIRUS (with BERT)
```bash
cd SIRUS/BERT/bash
bash fewrel_5shot.sh
bash tacred_5shot.sh
```

### Run SIRUS (with LLM)
```bash
cd SIRUS/LLM/bash
bash fewrel_5shot.sh
bash tacred_5shot.sh
```


## CED
### Change directory
```bash
cd CED
```
### Install requirements
```bash
pip install -r requirements.txt
```
### Run
```bash
bash sh/maven.sh
```
