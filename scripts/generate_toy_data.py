from __future__ import print_function
import argparse
import os
import shutil
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
    data_path = os.path.join(path, 'data.txt')
    with open(data_path, 'w') as fout:
        for _ in range(size):
            length = random.randint(1, args.max_len)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            fout.write("\t".join([" ".join(seq), " ".join(reversed(seq))]))
            fout.write('\n')

    # generate vocabulary
    src_vocab = os.path.join(path, 'vocab.source')
    with open(src_vocab, 'w') as fout:
        fout.write("\n".join([str(i) for i in range(10)]))
    tgt_vocab = os.path.join(path, 'vocab.target')
    shutil.copy(src_vocab, tgt_vocab)

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
