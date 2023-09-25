# SCEC2023-TeamH

## Team Information
| Name  | Affiliation | Email | Phone |
|-------|---------|------------------|------------------|
| Heehoon Kim* | Seoul National University | csehydrogen@gmail.com | 010-2569-3426 |
| Junyeol Ryu | Seoul National University | junyeol@aces.snu.ac.kr | 010-3329-7561 |
(* is the team lead)

## Repository Organization
```
.
├── llama_fast/          # LLaMA python package
│   ├── example.py/      #    - Main inference script
│   ├── model.py/        #    - LLaMA model components
│   ├── schedule.py/     #    - Batch scheduling module
│   ├── tokenizer.py/    #    - LLaMA tokenizer
```

## Setup
In order to mount model and tokenizer to docker, <DATA_DIR> should contain the following:
```
<DATA_DIR>
├── 30B_cpu_0.pth        # Horizontal partition 1 of LLaMA model checkpoint 
├── 30B_cpu_1.pth        # Horizontal partition 2 of LLaMA model checkpoint 
├── 30B_cpu_2.pth        # Horizontal partition 3 of LLaMA model checkpoint 
├── 30B_cpu_3.pth        # Horizontal partition 4 of LLaMA model checkpoint 
├── params.json          # Parameter metadata json file 
├── tokenizer.model      # Tokenizer checkpoint 
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