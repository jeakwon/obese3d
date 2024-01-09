import sys, argparse
from obese3d.utils import benchmark, argparser

if __name__ == '__main__':
    args = argparser()
    benchmark(args)