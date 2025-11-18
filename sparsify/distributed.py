import socket
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes


def dist_worker(
    worker: Callable,
    *worker_args,
):
    try:
        worker(*worker_args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Barrier failed during cleanup: {e}")
                pass

            dist.destroy_process_group()


def handle_distribute(process_name: str, worker, const_worker_args: list[Any]):
    """
    Launch a distributed multi-process job over all visible CUDA devices.

    Parameters
    ----------
    process_name : str
        Label used by Torch Elastic to tag logs and processes.
    worker : Callable
        Function that will be executed on every spawned process. It must accept
        ``(rank, world_size, *const_worker_args)`` in that order.
    const_worker_args : list
        Arguments passed verbatim to every worker invocation after ``rank`` and
        ``world_size``. These are typically configuration or shared datasets.
    """
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, *const_worker_args)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = None
        try:
            ctx = start_processes(
                process_name,
                dist_worker,
                args={
                    i: (worker, i, world_size, *const_worker_args)
                    for i in range(world_size)
                },
                envs={
                    i: {
                        "LOCAL_RANK": str(i),
                        "MASTER_ADDR": "localhost",
                        "MASTER_PORT": str(port),
                    }
                    for i in range(world_size)
                },
                logs_specs=DefaultLogsSpecs(),
            )
            ctx.wait()
        finally:
            if ctx is not None:
                ctx.close()  # Kill any processes that are still running
