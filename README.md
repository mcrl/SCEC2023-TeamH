# SCEC2023-TeamH

## Team Information
| Name  | Affiliation | Email | Phone |
|-------|---------|------------------|------------------|
| Heehoon Kim* | Seoul National University | csehydrogen@gmail.com | 010-2569-3426 |
| Junyeol Ryu | Seoul National University | jyeol.ryu@gmail.com | 010-3329-7561 |

(* is the team lead)

## Repository Organization
```
.
├── llama_fast/             # ⚡ LLaMA python package
│   ├── csrc/               #    - C extensions
│   ├── build.py            #    - Package build script
│   ├── example.py          #    - Main inference script
│   ├── model.py            #    - LLaMA model components
│   ├── schedule.py         #    - Batch scheduling module
│   ├── tokenizer.py        #    - LLaMA tokenizer
│   ├── run.sh              #    - Docker entry script 
├── tools/                  # 🛠️ LLaMA tools
│   ├── repartition_ckpt.py #    - Model ckpt repartition script
├── Dockerfile              # 🐳 LLaMA Docker build script
```

## Setup
Prepare <DATA_DIR> with the files from the original LLaMA 30B model checkpoint:
```
<DATA_DIR>
├── consolidated.00.pth     # Model parallel partition 0
├── consolidated.01.pth     # Model parallel partition 1
├── consolidated.02.pth     # Model parallel partition 2
├── consolidated.03.pth     # Model parallel partition 3
├── params.json             # Parameter metadata json file 
├── tokenizer.model         # Tokenizer checkpoint 
```

Then, execute the provided script to repartition the model checkpoint:
```
$ python tools/repartition_ckpt.py --data_dir <DATA_DIR>
```

If the repartition is successful, the <DATA_DIR> would contain the following additional files:
```
<DATA_DIR>
├── ...
├── 30B_cpu_0.pth           # Pipeline parallel partition 0
├── 30B_cpu_1.pth           # Pipeline parallel partition 1
├── 30B_cpu_2.pth           # Pipeline parallel partition 2
├── 30B_cpu_3.pth           # Pipeline parallel partition 3
├── ...
```

## How to Run
1. Build docker image
```bash
docker build -t <IMAGE_NAME> .
```
2. Run docker 
```bash
docker run --rm -it --ipc host --gpus all -v <DATA_DIR>:/data --name <CONTAINER_NAME> <IMAGE_NAME>
```
