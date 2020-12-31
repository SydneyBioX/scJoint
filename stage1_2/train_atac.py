from __future__ import print_function
import torch
from util.option import Options
from util.trainingprocess import TrainingProcess


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)

    model = TrainingProcess(args)
    model.load_checkpoint(args)

    if args.train_mode == 'all':
        for epoch in range(args.start_epoch, args.epochs + 1):
            print('Epoch:', epoch)
            model.train(epoch)
            model.test(epoch, args.train_mode)
    elif args.train_mode == 'test' or args.train_mode == 'test_print':
        model.test(args.start_epoch, args.train_mode)


if __name__ == "__main__":
    main()
