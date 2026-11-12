# SharpSeq + PRAGAS for Continual Event Detection (CED)

This folder applies PRAGAS (Guided SAM driven by distillation) on top of
SharpSeq (Le et al., 2024) for Continual Event Detection.
Data processing follows [Lifelong Event Detection with Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/) (Yu et al., EMNLP 2021).

## Requirements

```bash
pip install -r requirements.txt
```

## Data

No data is distributed with this repository. Obtain the datasets from their
official sources and put them under `data/`, so that you have `data/MAVEN/`
and `data/ACE/`:

- **MAVEN**: official release by THU-KEG (MAVEN-dataset).
- **ACE 2005**: Linguistic Data Consortium, catalog LDC2006T06. Requires an LDC license.

Preprocessing into the continual-learning streams follows
Yu et al. (EMNLP 2021) and SharpSeq (Le et al., 2024); see their official
repositories for the split scripts.

Preprocess MAVEN (extracts `bert-large-cased` features to `data/features/`):

```bash
python prepare_inputs.py
python prepare_stream_instances.py
```

`prepare_streams.py` builds `streams.json` (task splits) for a dataset.

## Training and evaluation

MAVEN (5 task permutations):

```bash
bash sh/maven.sh
```

Logs are written to `log/pragas_maven/perm_<i>/exp.log` (set `log_p` in `sh/maven.sh` to change it).

For ACE, run `run_train.py` with `--datasetname ACE --stream-file data/ACE/streams.json`
and the same PRAGAS options as in `sh/maven.sh` (`--sam 1 --sam-optimizer ... --rho ... --rho_weight ...`).

## Acknowledgements

The code is based on the official SharpSeq implementation.
