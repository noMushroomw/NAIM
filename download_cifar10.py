import argparse

import torchvision

def main():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--out', default='data/cifar10')
    args = p.parse_args()
    for train in (True, False):
        torchvision.datasets.CIFAR10(args.out, train=train, download=True)
    print(f'CIFAR-10 ready in {args.out}')

if __name__ == '__main__':
    main()
