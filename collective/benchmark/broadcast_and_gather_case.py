import argparse
import re

import ray
import ray.util.collective as col
import numpy as np
import time

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description=("None")
)

parser.add_argument(
    "--num-rounds",
    type=int,
    help="",
)

args, remaining_args = parser.parse_known_args()


@ray.remote(num_cpus=1)
class Sender:
    def __init__(self, world_size: int):
        self._world_size = world_size
        self.send_buf = np.zeros((2, 3), dtype=np.float64)
        col.init_collective_group(world_size, 0, backend="gloo", group_name="default")
        logger.warning("Sender, world_size is ", world_size)

    def do_compute(self):
        self.send_buf = np.random.randn(2,3)
        logger.warning("Starting broadcasting at time %s", time.time())
        col.broadcast(self.send_buf, 0)
        logger.warning("Ending broadcasting at time %s", time.time())
        for rank in range(1, self._world_size):
            recv_buf = np.zeros((2, 3), dtype=np.float64) 
            col.recv(recv_buf, rank)
        logger.warning("Ending recv at time %s", time.time()) 
        return True

    def destroy(self):
        col.destroy_group()

    def ping(self):
        return "pong"


@ray.remote(num_cpus=1)
class Recver:
    def __init__(self, world_size: int, curr_rank: int):
        self._rank = curr_rank
        self.recv_buf = np.zeros((2, 3), dtype=np.float64)
        col.init_collective_group(world_size, curr_rank, backend="gloo", group_name="default")

    def do_compute(self):
        logger.warning("Doing compute...")
        col.broadcast(self.recv_buf, 0, "default")
        logger.warning("Doing send...")
        col.send(self.recv_buf, 0)
        logger.warning("Finished compute at rank %s", self._rank)
        return True

    def destroy(self):
        col.destroy_group()

    def ping(self):
        return "pong"


def main(num_recvers: int):
    num_recvers = int(num_recvers)
    ray.init(num_cpus=6, ignore_reinit_error=True)

    sender = Sender.remote(num_recvers + 1)
    all_recvers = []
    for i in range(0, num_recvers):
        recver = Recver.options(lifetime="detached").remote(num_recvers + 1, i + 1)
        all_recvers.append(recver)

    ray.get(sender.ping.remote())
    for recver in all_recvers:
        ray.get(recver.ping.remote())

    # Do computing
    recv_objs = []
    for recver in all_recvers:
        recv_obj = recver.do_compute.remote()
        recv_objs.append(recv_obj)
    send_obj = sender.do_compute.remote()

    print("send_buf:", ray.get(send_obj))
    for recv_obj in recv_objs:
       print("recv_buf:", ray.get(recv_obj)) 
    print("==========END")


if __name__ == "__main__":
    num_rounds = args.num_rounds
    ray.init(num_cpus=6, ignore_reinit_error=True)
    main(int(num_rounds))
