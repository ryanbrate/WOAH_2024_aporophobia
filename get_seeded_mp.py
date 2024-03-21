"""
get seeded over multiple (compressed) files, via multi-processing
retrieve comments for all seeds and a neutral (non rich/poor) sample

"""

import bz2
import json
import pathlib
import re
from collections import defaultdict
from itertools import cycle
from functools import reduce
from operator import concat
import numpy as np

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def main():

    # import the config
    with open("config.json", 'r') as f:
        config = json.load(f)

    np.random.seed(config['random_seeds']['numpy'])
    fps:list[pathlib.Path] = [pathlib.Path(fp_str).expanduser().resolve() for fp_str in config['fps_str']]


    # init containers
    comments = defaultdict(list)
    indices = defaultdict(list)  # where indices[seed][i] corresponds to comments[seed][i]
    counts = []

    # gather comments
    for comments_, indices_, count_ in process_map(
        get_seeded_star, zip(fps, range(len(fps)), cycle([config]))
    ):
        for s,c in comments_.items():
            comments[s] += c
        for s,i in indices_.items():
            indices[s] += i
        counts.append(count_)
        
    # save ...
    with open("comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f)
    with open("indices.json", "w", encoding="utf-8") as f:
        json.dump(indices, f)
    with open("count.json", "w", encoding="utf-8") as f:
        json.dump(list(zip(config['fps_str'], counts)), f)


def get_seeded_star(t):
    return get_seeded(*t)


def get_seeded(cf_path: str, cf_i: int, config:dict) -> dict:
    """Return dict of indices wrt., cf_path for each group

    Args:
        cf_path (str): path to bzip2 file
        cf_i (int): integer denoting compressed file index
    """

    
    # get the user-specified config variables
    wealth_seeds:list[str] = reduce(concat, [config['wealth_seeds'][key] for key in config['wealth_seeds'].keys()])
    other_seeds:list[str] = reduce(concat, [config['other_seeds'][key] for key in config['other_seeds'].keys()])
    p_neutral = config['p_neutral']

    # init. the returned
    comments = defaultdict(list)
    indices = defaultdict(list)


    ## iterate over each comment in compressed file
    with bz2.BZ2File(cf_path) as cf:
        for i, line in tqdm(enumerate(cf, start=1), desc=f"cf_i={cf_i}"):
            comment: str = json.loads(line)["body"]

            # get indices and comments wrt., wealth seeds
            neutral_candidate = True
            for seed in wealth_seeds:

                # get wealth seed matches
                if contains(seed, comment):
                    indices[seed].append((cf_i, i))
                    comments[seed].append(comment)
                    neutral_candidate = False

            # get indices and comments wrt., other seeds
            for seed in other_seeds:
                if contains(seed, comment):
                    indices[seed].append((cf_i, i))
                    comments[seed].append(comment)

            # flip a coin to sample as a neutral wealth context if not containing a seed word from wealth_seeds
            if neutral_candidate:
                if np.random.binomial(1, config['p_neutral']) == 1:
                    comments['neutral_sample'].append(comment)
                    indices['neutral_sample'].append((cf_i, i))

    return comments, indices, i


def contains(seed, comment)->bool:
    """Return True if seed in comment"""

    # crude search filter
    if seed in comment or seed.capitalize() in comment:

        # fine-grained search
        found = re.search(rf"\b{seed}\b", comment, re.IGNORECASE)
        if found is not None:
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    main()
