# Meghnad - The core ML library

## Copyright
All rights reserved. Inxite Out Pvt Ltd. 2022.

## Author
Kaushik Bar (kaushik.bar@inxiteout.ai)



## Setup Instructions
conda create -n "your-env-name" python=3.9.12 anaconda

conda activate "your-env-name"

conda install pytorch=1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch

cd "parent directory of meghnad"

### Package
pip install -e .

### NLP
pip install -r meghnad/interface/nlp/requirements.txt

python -m spacy en_core_web_sm
python -m spacy en_core_web_md
python -m spacy en_core_web_lg
python -m spacy en_core_web_trf

### CV
pip install -r meghnad/interface/cv/requirements.txt

### STD ML
pip install -r meghnad/interface/std_ml/requirements.txt

### TS
pip install -r meghnad/interface/ts/requirements.txt

### Speech
pip install -r meghnad/interface/speech/requirements.txt

