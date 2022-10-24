# AI Accelerator powered by Meghnad

## Copyright
All rights reserved. Inxite Out Pvt Ltd. 2022.

## Author
Kaushik Bar (kaushik.bar@inxiteout.ai)



## Setup Instructions
conda create -n "your-env-name" python=3.9.12 anaconda

conda activate "your-env-name"

conda install pytorch=1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch

cd "root directory ('ixolerator')"

### Package
pip install -e .

### Apps
pip install -r apps/requirements.txt

### Meghnad

#### NLP
pip install -r meghnad/interfaces/nlp/requirements.txt

#### CV
pip install -r meghnad/interfaces/cv/requirements.txt

#### STD ML
pip install -r meghnad/interfaces/std_ml/requirements.txt

#### TS
pip install -r meghnad/interfaces/ts/requirements.txt

#### Speech
pip install -r meghnad/interfaces/speech/requirements.txt

### Connectors

#### AWS
chmod a+x connectors/aws/script.sh

bash connectors/aws/script.sh

pip install -r connectors/aws/requirements.txt

