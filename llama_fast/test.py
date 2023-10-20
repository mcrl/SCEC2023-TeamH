import torch
import torch.distributed as dist
import os
import sys
import teamh_c_helper
import logging
import time

logger = logging.getLogger('llama_fast')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(process)d][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))

torch.distributed.init_process_group("nccl")
torch.cuda.set_device(local_rank)

teamh_c_helper.init(local_rank, world_size)
torch.distributed.barrier() # barrier is necessary to ensure semaphore is initialized
teamh_c_helper.init_comm()

#if local_rank == 0:
#  teamh_c_helper.test()


niter = 10
B, S = 8000, 8000
h = torch.rand((B, S), dtype=torch.float32, device='cuda')

for i in range(niter):
  torch.cuda.synchronize()
  st = time.time()

  if local_rank == 0:
    logger.info(f'Rank {local_rank} iter {i} Before send {h.flatten()[0]}')
    teamh_c_helper.send(h, i == 0)

  if local_rank == 1:
    #teamh_c_helper.recv(h)
    logger.info(f'Rank {local_rank} iter {i} After recv {h.flatten()[0]}')

  torch.cuda.synchronize()
  et = time.time()
  if et > st:
    gbps = (B * S * 4) / (et - st) / 1e9
    logger.info(f'Rank {local_rank} iter {i} took {(et - st)} seconds {gbps} GB/s')

  #if local_rank == 2:
  #  h = torch.rand((B, S), dtype=torch.float32, device='cuda')
  #  teamh_c_helper.recv(h)
  #  teamh_c_helper.send(h, i == 0)
  #  logger.info(f'Rank {local_rank} iter {i} After recv {h.flatten()[0]}')

  #if local_rank == 3:
  #  h = torch.rand((B, S), dtype=torch.float32, device='cuda')
  #  teamh_c_helper.recv(h)
  #  logger.info(f'Rank {local_rank} iter {i} After recv {h.flatten()[0]}')

torch.distributed.barrier() # barrier is necessary so that semaphore is not destroyed too early (i.e. while other processes are still using it)
teamh_c_helper.finalize()

#if local_rank == 0:
#  grp0 = torch.distributed.new_group([0, 1])
#if local_rank == 1:
#  grp0 = torch.distributed.new_group([0, 1])
#if local_rank == 2:
#  grp0 = torch.distributed.new_group([2, 3])
#if local_rank == 3:
#  grp0 = torch.distributed.new_group([2, 3])
#
#if local_rank == 0:
#  grp1 = torch.distributed.new_group([0, 3])
#if local_rank == 1:
#  grp1 = torch.distributed.new_group([1, 2])
#if local_rank == 2:
#  grp1 = torch.distributed.new_group([1, 2])
#if local_rank == 3:
#  grp1 = torch.distributed.new_group([0, 3])
#
#if local_rank == 0:
#  grp2 = torch.distributed.new_group([0, 1])
#if local_rank == 1:
#  grp2 = torch.distributed.new_group([0, 1])
#if local_rank == 2:
#  grp2 = torch.distributed.new_group([2, 3])
#if local_rank == 3:
#  grp2 = torch.distributed.new_group([2, 3])
#
#if local_rank == 0:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.isend(h, local_rank + 1, group = grp0)
#  handle = dist.broadcast(h, local_rank, group = grp0, async_op = True)
#  handle.wait()
#  print(f'Rank {local_rank} done sending')
#
#if local_rank == 1:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.irecv(h, local_rank - 1, group = grp0)
#  handle = dist.broadcast(h, local_rank - 1, group = grp0, async_op = True)
#  handle.wait()
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.isend(h, local_rank + 1, group = grp1)
#  handle = dist.broadcast(h, local_rank, group = grp1, async_op = True)
#  handle.wait()
#  print(f'Rank {local_rank} done sending')
#
#if local_rank == 2:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.irecv(h, local_rank - 1, group = grp1)
#  handle = dist.broadcast(h, local_rank - 1, group = grp1, async_op = True)
#  handle.wait()
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.isend(h, local_rank + 1, group = grp2)
#  handle = dist.broadcast(h, local_rank, group = grp2, async_op = True)
#  handle.wait()
#  print(f'Rank {local_rank} done recving')
#
#if local_rank == 3:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.irecv(h, local_rank - 1, group = grp2)
#  handle = dist.broadcast(h, local_rank - 1, group = grp2, async_op = True)
#  handle.wait()
#  print(f'Rank {local_rank} done recving')

#sendgrp, recvgrp = None, None
#sendgrp_peer, recvgrp_peer = None, None
#for i in range(1, world_size):
#  if local_rank == i - 1:
#    print(f'Rank {local_rank} building sendgrp {local_rank} {local_rank + 1}')
#    sendgrp = torch.distributed.new_group([local_rank, local_rank + 1])
#    sendgrp_rank = torch.distributed.get_group_rank(sendgrp, local_rank)
#    sendgrp_peer = 1 - sendgrp_rank
#    print(f'Rank {local_rank} sendgrp_peer {sendgrp_peer}')
#  if local_rank == i:
#    print(f'Rank {local_rank} building recvgrp {local_rank - 1} {local_rank}')
#    recvgrp = torch.distributed.new_group([local_rank - 1, local_rank])
#    recvgrp_rank = torch.distributed.get_group_rank(recvgrp, local_rank)
#    recvgrp_peer = 1 - recvgrp_rank
#    print(f'Rank {local_rank} recvgrp_peer {recvgrp_peer}')
#
## seed must be the same in all processes
#torch.manual_seed(1)
#
#if local_rank == 0:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  handle = dist.isend(h, local_rank + 1, group = sendgrp)
#  handle.wait()
#  print(f'Rank {local_rank} done sending')
#
#if local_rank == 1:
#  h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  handle = dist.irecv(h, local_rank - 1, group = recvgrp)
#  handle.wait()
#  #h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.isend(h, local_rank + 1, group = sendgrp)
#  #handle.wait()
#  print(f'Rank {local_rank} done sending')
#
#if local_rank == 2:
#  #h = torch.empty((2, 2), dtype=torch.float32, device='cuda')
#  #handle = dist.irecv(h, local_rank - 1, group = recvgrp)
#  #handle.wait()
#  print(f'Rank {local_rank} done recving')
#
#print(f'Rank {local_rank} done waiting')
#