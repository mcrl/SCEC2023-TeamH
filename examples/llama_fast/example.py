import time
time_py_entered = time.time()

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple
from multiprocessing import Process, Queue
import os
from datasets import load_dataset
import numpy as np
import torch
from transformers import LlamaTokenizer

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

import schedule
import cProfile
import teamh_c_helper

NUM_CHOICES = 4
DATA_LIMIT = 22222
CTX_GRP_LIMIT = 22222
CTX_THR = 10000
CTX_MINIBATCH_THR = 2048
CONT_THR = 2048
USE_CUSTOM_COMM = True
DEBUG_SCHEDULE = False
DEBUG_ANSWER = False
DEBUG_PERFORMANCE = False
DEBUG_PROFILE = False
DEBUG_PYTHON_PROFILE = False
DEBUG_COMPARE_GOLD = False
DEBUG_TURN_OFF_COMM = False


def read_config(config_path: Path):
  with open(config_path, 'r') as f:
      config = json.load(f)
  use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
  remove_input_padding = config['plugin_config']['remove_input_padding']
  dtype = config['builder_config']['precision']
  tp_size = config['builder_config']['tensor_parallel']
  pp_size = config['builder_config']['pipeline_parallel']
  world_size = tp_size * pp_size
  assert world_size == tensorrt_llm.mpi_world_size(), \
      f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
  num_heads = config['builder_config']['num_heads'] // tp_size
  hidden_size = config['builder_config']['hidden_size'] // tp_size
  vocab_size = config['builder_config']['vocab_size']
  num_layers = config['builder_config']['num_layers']
  num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
  paged_kv_cache = config['plugin_config']['paged_kv_cache']
  tokens_per_block = config['plugin_config']['tokens_per_block']
  quant_mode = QuantMode(config['builder_config']['quant_mode'])
  if config['builder_config'].get('multi_query_mode', False):
      tensorrt_llm.logger.warning(
          "`multi_query_mode` config is deprecated. Please rebuild the engine."
      )
      num_kv_heads = 1
  num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
  use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                      False)

  model_config = ModelConfig(num_heads=num_heads,
                              num_kv_heads=num_kv_heads,
                              hidden_size=hidden_size,
                              vocab_size=vocab_size,
                              num_layers=num_layers,
                              gpt_attention_plugin=use_gpt_attention_plugin,
                              paged_kv_cache=paged_kv_cache,
                              tokens_per_block=tokens_per_block,
                              remove_input_padding=remove_input_padding,
                              dtype=dtype,
                              quant_mode=quant_mode,
                              use_custom_all_reduce=use_custom_all_reduce)

  return model_config, tp_size, pp_size, dtype

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_output_len', type=int, required=True)
  parser.add_argument('--log_level', type=str, default='error')
  parser.add_argument('--engine_dir', type=str, default='llama_outputs')
  parser.add_argument('--tokenizer_dir',
                      type=str,
                      default=".",
                      help="Directory containing the tokenizer.model.")
  parser.add_argument('--cache_dir',
                      type=str,
                      default=".",
                      help="Directory containing the dataset cache.")
  parser.add_argument('--input_text',
                      type=str,
                      default='Born in north-east France, Soyer trained as a')
  parser.add_argument(
      '--input_tokens',
      dest='input_file',
      type=str,
      help=
      'CSV or Numpy file containing tokenized input. Alternative to text input.',
      default=None)
  parser.add_argument('--output_csv',
                      type=str,
                      help='CSV file where the tokenized output is stored.',
                      default=None)
  parser.add_argument('--output_npy',
                      type=str,
                      help='Numpy file where the tokenized output is stored.',
                      default=None)
  parser.add_argument('--num_beams',
                      type=int,
                      help="Use beam search if num_beams >1",
                      default=1)
  parser.add_argument('--streaming', default=False, action='store_true')
  parser.add_argument('--streaming_interval',
                      type=int,
                      help="How often to return tokens when streaming.",
                      default=5)
  return parser.parse_args()


def main(
  max_output_len: int,
  log_level: str = 'error',
  engine_dir: str = 'llama_outputs',
  input_text: str = 'Born in north-east France, Soyer trained as a',
  input_file: str = None,
  output_csv: str = None,
  output_npy: str = None,
  tokenizer_dir: str = None,
  cache_dir: str = None,
  num_beams: int = 1,
  streaming: bool = False,
  streaming_interval: int = 5,
):
  tensorrt_llm.logger.set_level(log_level)
  print(f'[Rank Unknown] main entered: {time.time() - time_py_entered:.2f} seconds')

  engine_dir = Path(engine_dir)
  config_path = engine_dir / 'config.json'
  model_config, tp_size, pp_size, dtype = read_config(config_path)
  world_size = tp_size * pp_size

  local_rank = tensorrt_llm.mpi_rank()
  runtime_mapping = tensorrt_llm.Mapping(world_size,
                                          local_rank,
                                          tp_size=tp_size,
                                          pp_size=pp_size)
  torch.cuda.set_device(local_rank % runtime_mapping.gpus_per_node)

  """================
  Tokenizer Disk I/O
  ================"""
  tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)
  print(f'[Rank {local_rank}] tokenizer loading done: {time.time() - time_py_entered:.2f} seconds')

  # Load and preprocess dataset with multiprocessing
  def _load_preprocess_and_schedule_dataset(q, cache_dir, tokenizer):
    """================
    Dataset Disk I/O
    ================"""
    dataset = load_dataset("hellaswag", cache_dir=cache_dir, split='validation')
    print(f'[Rank {local_rank}] dataset loading done: {time.time() - time_py_entered:.2f} seconds')

    """================
    Dataset Tokenization and Scheduling
    ================"""
    docs, tokenized_docs, batches = schedule.preprocess_and_schedule_dataset(dataset, tokenizer, DATA_LIMIT, CTX_THR, CTX_MINIBATCH_THR, CONT_THR)
    max_bs, max_b, _ = schedule.max_sizes_of_batches(batches)
    q.put((docs, tokenized_docs, batches, max_bs, max_b))
    print(f'[Rank {local_rank}] preprocess_and_schedule_dataset done: {time.time() - time_py_entered:.2f} seconds')

  # As (the dataset preprocess time) ~ (model checkpoint load time), we overlap them
  q = Queue()
  p = Process(target=_load_preprocess_and_schedule_dataset, args=(q, cache_dir, tokenizer))
  p.start()

  """================
  Model Checkpoint Disk I/O + Send to GPU
  ================"""
  engine_name = get_engine_name('llama', dtype, tp_size, pp_size, local_rank)
  serialize_path = engine_dir / engine_name
  with open(serialize_path, 'rb') as f:
      engine_buffer = f.read()
  decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                    engine_buffer,
                                                    runtime_mapping,
                                                    debug_mode=False,
                                                    debug_tensors_to_save=None)
  decoder.setup(input_lengths.size(0), max_input_length, max_output_len, num_beams)
  print(f'[Rank {local_rank}] checkpoint loading done: {time.time() - time_py_entered:.2f} seconds')

  """================
  Join with dataset process
  ================"""
  torch.cuda.nvtx.range_push('Join dataset process')
  docs, tokenized_docs, batches, max_bs, max_b = q.get()
  p.join()
  torch.cuda.nvtx.range_pop()
  print(f'[Rank {local_rank}] dataset process joining done: {time.time() - time_py_entered:.2f} seconds')


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
