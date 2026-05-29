"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import pprint
import time
import lmdb
import os
from tqdm import tqdm

import synthtiger

def writeCache(env, cache):
    """Write the cache to the LMDB database."""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def run(args):
    if args.config is not None:
        config = synthtiger.read_config(args.config)

    pprint.pprint(config)

    synthtiger.set_global_random_seed(args.seed)
    template = synthtiger.read_template(args.script, args.name, config)
    generator = synthtiger.generator(
        args.script,
        args.name,
        config=config,
        count=args.count,
        worker=args.worker,
        seed=args.seed,
        retry=True,
        verbose=args.verbose,
    )

    
    if args.lmdb:
        os.makedirs(args.output, exist_ok=True)
        env = lmdb.open(args.output, map_size=1099511627776)
        cache = {}
        cnt = 1
        for idx, (task_idx, data) in tqdm(enumerate(generator)):
            cache, cnt = template.save_lmdb(root=args.output, data=data, env=env, cache=cache, cnt=cnt)
        cache["num-samples".encode()] = str(cnt - 1).encode()
        writeCache(env, cache)
        print(f"Created LMDB dataset with {cnt - 1} samples.")

    else:
        if args.output is not None:
            template.init_save(args.output)
        for idx, (task_idx, data) in enumerate(generator):
            if args.output is not None:
                template.save(args.output, data, task_idx)
            print(f"Generated {idx + 1} data (task {task_idx})")
        if args.output is not None:
            template.end_save(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        help="Directory path to save data.",
    )
    parser.add_argument(
        "-c",
        "--count",
        metavar="NUM",
        type=int,
        default=100,
        help="Number of output data. [default: 100]",
    )
    parser.add_argument(
        "-w",
        "--worker",
        metavar="NUM",
        type=int,
        default=0,
        help="Number of workers. If 0, It generates data in the main process. [default: 0]",
    )
    parser.add_argument(
        "-s",
        "--seed",
        metavar="NUM",
        type=int,
        default=None,
        help="Random seed. [default: None]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print error messages while generating data.",
    )
    parser.add_argument(
        "script",
        metavar="SCRIPT",
        type=str,
        help="Script file path.",
    )
    parser.add_argument(
        "name",
        metavar="NAME",
        type=str,
        help="Template class name.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        type=str,
        nargs="?",
        help="Config file path.",
    )
    parser.add_argument(
        "--lmdb",
        action="store_true",
        help="Save as lmdb.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
