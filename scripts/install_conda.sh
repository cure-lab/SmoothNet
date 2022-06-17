export CONDA_ENV_NAME=smoothnet-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.6

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt