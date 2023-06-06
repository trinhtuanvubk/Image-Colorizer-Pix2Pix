import argparse

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    # scenario
    parser.add_argument('--scenario', type=str, default='main')

    # parse args
    args = parser.parse_args()
    return args

def print_args(args):
    pass