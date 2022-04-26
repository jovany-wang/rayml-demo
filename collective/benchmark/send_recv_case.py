import argparse

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
    def __init__(self):
        self.send_buf = np.zeros((2, 3), dtype=np.float64)
        col.init_collective_group(2, 0, backend="gloo", group_name="default")

    def do_send(self, num_rounds: int):
        start_time = time.time()
        logger.warning("Starting to send num_round(%s) at time %s", num_rounds, start_time)
        for _ in range(0, num_rounds):
            self.send_buf = np.random.randn(2,3)       
            col.send(self.send_buf, 1, group_name="default")
        end_time = time.time()
        logger.warning("End to send at time %s", end_time)
        return True

    def destroy(self):
        col.destroy_group()

    def ping(self):
        return "pong"


@ray.remote(num_cpus=1)
class Recver:
    def __init__(self):
        self.recv_buf = np.zeros((2, 3), dtype=np.float64)
        col.init_collective_group(2, 1, backend="gloo", group_name="default")

    def do_recv(self, num_rounds: int):
        start_time = time.time()
        logger.warning("Starting to recv  num_round(%s) at time %s", num_rounds, start_time)
        for i in range(0, num_rounds):
          col.recv(self.recv_buf, 0, group_name="default")
          if i % 1000 == 0:
            logger.warning("Received ndarr is %s, current round is %d", self.recv_buf, i)
        end_time = time.time()
        logger.warning("End to recv at time %s", end_time)
        return True

    def destroy(self):
        col.destroy_group()

    def ping(self):
        return "pong"


def main(num_rounds: int):
    num_rounds = int(num_rounds)
    ray.init(num_cpus=6, ignore_reinit_error=True)
    logger.warning("Received num_rounds: %s", num_rounds)
    print("Received num_rounds: {}".format(num_rounds))
    sender = Sender.remote()
    recver = Recver.remote()

    ray.get(sender.ping.remote())
    ray.get(recver.ping.remote())

    send_obj = sender.do_send.remote(num_rounds)
    recv_obj = recver.do_recv.remote(num_rounds)

    print("send_buf:", ray.get(send_obj))
    print("recv_buf:", ray.get(recv_obj))
    print("==========END")


if __name__ == "__main__":
    num_rounds = args.num_rounds
    ray.init(num_cpus=6, ignore_reinit_error=True)
    main(int(num_rounds))
