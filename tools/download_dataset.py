import torch
from datasets import load_dataset
import fire
import pickle

def main(cache_dir):
  dataset = load_dataset("hellaswag", cache_dir=cache_dir, split='validation')
  pickle.dump(dataset, open('hellaswag_validation.pkl', 'wb'))

if __name__ == '__main__':
  fire.Fire(main)