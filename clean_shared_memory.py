from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from time import sleep, time
from typing import List, Union

import torch.distributed as dist

from streaming.base.constant import SHM_TO_CLEAN
from streaming.base.distributed import get_local_rank, maybe_init_dist
from streaming.base.shared.prefix import _get_path
import streaming
import composer
import logging

# The streaming module uses inter-process shared memory (https://docs.python.org/3/library/multiprocessing.shared_memory.html).
# If for any reason a process using this module needs to be manually killed (e.g. with kill -9),
# then it will not clean up the shared memory.
# The next time you launch the training run, it will notice that the shared memory is already in use.
# So, we clean it up by just iterating through all the possible shared memory locations and unlinking them.
# This will cause a problem if you try to run two different jobs in parallel on the same machine.
# I'm going to assume that that is unlikely.
# The function below is mostly copied directly from 
# https://github.com/mosaicml/streaming/blob/main/streaming/base/util.py
# THe only alterations are:
# 1. including a sleep(1) on the non-leader processes that seems to mitigate a race
# condition.
# 2. logging results when memory is cleaned.
def clean_stale_shared_memory_verbose() -> None:
    """Clean up all the leaked shared memory.

    In case of a distributed run, clean up happens on local rank 0 while other local ranks wait for
    the local rank 0 to finish.
    """
    # Initialize torch.distributed ourselves, if necessary.
    destroy_dist = maybe_init_dist()
    
    if composer.utils.dist.get_local_rank() != 0:
        sleep(1)

    # Perform clean up on local rank 0
    if get_local_rank() == 0:
        any_leaked_shm = False
        for prefix_int in range(1000000):
            leaked_shm = False
            for shm_name in SHM_TO_CLEAN:
                name = _get_path(prefix_int, shm_name)
                try:
                    shm = BuiltinSharedMemory(name, True, 4)
                except FileExistsError:
                    shm = BuiltinSharedMemory(name, False, 4)
                    any_leaked_shm = True
                    leaked_shm = True
                finally:
                    if leaked_shm:
                        logging.info(f"cleaning up some memory: {name}")
                    shm.close()  # pyright: ignore
                    shm.unlink()
            # Come out of loop if no leaked shared memory
            if not leaked_shm:
                break
        if not any_leaked_shm:
            logging.info(f"there was NO leaked shared memory!")

    # Sync all ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Delete the process group if Streaming initialized it.
    if destroy_dist:
        dist.destroy_process_group()

clean_stale_shared_memory_verbose()