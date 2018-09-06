from __future__ import print_function
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="../data")
parser.add_argument('--max-len', help="max sequence length", default=10)
args = parser.parse_args()


def generate_dataset(root, name, size):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)

    # generate data file
    src_path = os.path.join(path, 'src.txt')
    tgt_path = os.path.join(path, 'tgt.txt')
    with open(src_path, 'w') as src_out, open(tgt_path, 'w') as tgt_out:
        for _ in range(size):
            length = random.randint(1, args.max_len)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            src_out.write(" ".join(seq) + "\n")
            tgt_out.write(" ".join(reversed(seq)) + "\n")


if __name__ == '__main__':
    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    toy_dir = os.path.join(data_dir, 'toy_reverse')
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    generate_dataset(toy_dir, 'train', 10000)
    generate_dataset(toy_dir, 'dev', 1000)
    generate_dataset(toy_dir, 'test', 1000)
