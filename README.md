## Installation Instruction
```bash
conda install -c -forge cudatoolkit-dev -y
conda create -n isft python==3.9
conda activate isft
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
cd ISFT
pip3 install -e .
pip install wandb IPython matplotlib
``` 
```bash
echo export HF_HOME="/scratch/user_name/huggingface" >> ~/.bashrc
echo export HF_DATASETS_CACHE="/scratch/user_name/huggingface/datasets" >> ~/.bashrc
echo export TRANSFORMERS_CACHE="/scratch/user_name/huggingface/models" >> ~/.bashrc
echo export HUGGINGFACE_HUB_CACHE="/scratch/user_name/huggingface/hub" >> ~/.bashrc
source ~/.bashrc
```

## Running ISFT

```bash
python3 examples/data_preprocess/countdown.py --local_dir ~/data/countdown3 --data_source countdown3
python3 examples/data_preprocess/countdown.py --local_dir ~/data/countdown4 --data_source countdown4
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
bash scripts/train_tiny_zero_isft.sh
```

## Running GRPO

```bash
bash scripts/train_tiny_zero_grpo.sh
```
