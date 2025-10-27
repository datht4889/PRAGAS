# SharpSeq: Empowering Continual Event Detection through Sharpness-Aware Sequential-task Learning
Source code for the ACL Rolling Review submission SharpSeq.


## Data & Model Preparation
To access the dataset, please visit: https://drive.google.com/drive/folders/10eQsBwqXSGkuh9pZ_X_6fQsKG_UDIaPH


To preprocess the data similar to [Lifelong Event Detection with Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/) (Yu et al., EMNLP 2021), run the following commands:
```bash
python prepare_inputs.py
python prepare_stream_instances.py
```

## Training and Testing

To start training and testing on MAVEN, run:
```bash
sh sh/maven.sh
```

## Requirements:
- transformer==4.52.4
- torch==2.6.0+cu124
- numpy==1.26.4
- tqdm==4.64.1
- scikit-learn==1.2.2
- cvxpy==1.6.6

