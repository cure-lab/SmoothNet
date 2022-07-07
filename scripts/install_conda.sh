export CONDA_ENV_NAME=smoothnet-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.6

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

pip install -r requirements.txt