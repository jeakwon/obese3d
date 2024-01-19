import sys
import argparse
from utils import benchmark

def argparser():
    # Creating an argument parser
    parser = argparse.ArgumentParser(description='Training Benchmark')

    # Adding arguments
    parser.add_argument('--model_name', type=str, default='LSTMClassifier', help='Model name')
    parser.add_argument('--model_kwargs', type=str, default="{}", help='Additional keyword arguments for the model')

    parser.add_argument('--criterion_name', type=str, default='CrossEntropyLoss', help='Name of the loss criterion')
    parser.add_argument('--criterion_kwargs', type=str, default="{}", help='Additional keyword arguments for the criterion')

    parser.add_argument('--optimizer_name', type=str, default='AdamW', help='Name of the optimizer')
    parser.add_argument('--optimizer_kwargs', type=str, default="{}", help='Additional keyword arguments for the optimizer')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')

    parser.add_argument('--data_dir', type=str, default='../coord', help='Directory for the dataset')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--num_joints', type=int, default=9, help='Number of joints')
    parser.add_argument('--dimension', type=int, default=3, help='Dimension of the input')

    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for the model')
    parser.add_argument('--output_size', type=int, default=24, help='Output size of the model')
    parser.add_argument('--target_type', type=str, default='ID', help='Target type for the dataset')

    parser.add_argument('--save_dir', default='./test',  type=str, help='Directory to save the model state')
    parser.add_argument('--quiet', action='store_true', help='suppress print logs')

    # Parsing arguments
    args = parser.parse_args()
    return args

def benchmark_notebook(prompt):
    sys.argv = prompt.split(' ')
    args = argparser()
    benchmark(args)
    
if __name__ == '__main__':
    args = argparser()
    benchmark(args)