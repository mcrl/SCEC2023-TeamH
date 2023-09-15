# Setup conda with python 3 and pytorch
conda create env -n scec python=3
conda activate scec
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install llama_v1 deps
pip install fairscale fire sentencepiece
# Install llama_v1
pip install -e llama-llama_v1

# Setup local disk on a0
fdisk /dev/nvme1n1
# Type: n, p, 1, and defaults for other options, then w
mkfs -t ext4 /dev/nvme1n1p1
mkdir local_disk
mount /dev/nvme1n1p1 local_disk
chown -R heehoon local_disk

# Copy pretrained llama to local disk
mkdir local_disk/llama_pretrained
cp -r ~/llama/llama_pretrained/tokenizer.model local_disk/llama_pretrained
cp -r ~/llama/llama_pretrained/30B local_disk/llama_pretrained

# For loading hellaswag
pip install datasets