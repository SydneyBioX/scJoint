import torch
import torch.optim as optim
from torch.autograd import Variable

from util.gene_in_metadata import Dataloader
from util.model_regress import Net_encoder, Net_cell
from util.closs import L1regularization, CellLoss, EncodingLoss, CenterLoss
from util.utils import *


def prepare_input(data_list):
    output = []
    for data in data_list:
        output.append(Variable(data.cuda()))
    return output


class TrainingProcess():
    def __init__(self, args):
        self.model_encoder = torch.nn.DataParallel(Net_encoder(args).cuda())
        self.model_cell = torch.nn.DataParallel(Net_cell(args).cuda())
        self.train_loader, self.test_loader = Dataloader(args).getloader()
        self.criterion_cell = CellLoss()
        self.criterion_center = CenterLoss()
        self.criterion_encoding = EncodingLoss(dim=64, p=args.encoding_p)
        self.l1_regular = L1regularization()

        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=args.lr, momentum=args.momentum,
                                           weight_decay=args.weight_decay)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        self.args = args
        self.best_pred = 100.

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * (0.1 ** ((epoch - 1) // self.args.lr_decay))
        if (epoch - 1) % self.args.lr_decay == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def load_checkpoint(self, args):
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
                self.best_pred = checkpoint['best_pred']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no resume checkpoint found at '{}'".format(args.resume))

    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        total_encoding_loss, total_cell_loss, total_sample_loss, total_kl_loss = 0., 0., 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        for batch_idx, (in_atac_data, in_rna_data, atac_label, rna_label) in enumerate(
                self.train_loader):  # train_loader
            # prepare data
            atac_data, rna_data, atac_label, rna_label = prepare_input(
                [in_atac_data, in_rna_data, atac_label, rna_label])

            # model forward
            atac_embedding, rna_embedding = self.model_encoder(atac_data, rna_data)
            atac_cell_prediction, rna_cell_prediction = self.model_cell(atac_embedding, rna_embedding)

            cell_loss = self.criterion_cell(rna_cell_prediction, rna_label) + self.criterion_cell(atac_cell_prediction, atac_label)
            encoding_loss = self.criterion_encoding(atac_embedding, rna_embedding)
            center_loss = self.criterion_center(atac_embedding, atac_label) + self.criterion_center(rna_embedding, rna_label)
			
            # update encoding weights
            regularization_loss = self.l1_regular(self.model_encoder)
            self.optimizer_encoder.zero_grad()
            regularization_loss.backward(retain_graph=True)           
            cell_loss.backward(retain_graph=True)
            encoding_loss.backward(retain_graph=True)
            center_loss.backward(retain_graph=True)
            self.optimizer_encoder.step()

            # update cell weights
            regularization_loss = self.l1_regular(self.model_cell)
            self.optimizer_cell.zero_grad()
            cell_loss.backward(retain_graph=True)
            regularization_loss.backward()
            self.optimizer_cell.step()

            # print log
            total_encoding_loss += encoding_loss.data.item()
            total_cell_loss += cell_loss.data.item()
            total_kl_loss += center_loss.data.item()

            progress_bar(batch_idx, len(self.train_loader),
                         'encoding_loss: %.3f, rna_loss: %.3f, center_loss: %.3f ' % (
                             total_encoding_loss / (batch_idx + 1), total_cell_loss / (batch_idx + 1),
                             total_kl_loss / (batch_idx + 1)))

    def test(self, epoch, train_mode):
        self.model_cell.eval()
        self.model_encoder.eval()

        atac_cell_correct, atac_cell_total = 0, 0
        is_best = False
        if self.args.train_mode == 'test_print':
            fp_em = open('./output_txt/'+str(self.args.subsample_col)+'/' + self.args.rnaoratac + '_embeddings.txt', 'w')
            fp_pre = open('./output_txt/'+str(self.args.subsample_col)+'/' + self.args.rnaoratac + '_predictions.txt', 'w')

        for batch_idx, (data, cell_target) in enumerate(self.test_loader):  # train_loader
            # prepare data
            data, cell_target = prepare_input([data, cell_target])

            # forward
            atac_embedding, rna_embedding = self.model_encoder(data, data)
            atac_cell_prediction, rna_cell_prediction = self.model_cell(atac_embedding, rna_embedding)
                                 
            # record features(you can modify this block for recording predictions)
            if self.args.rnaoratac == 'atac':
                embedding = atac_embedding.data.cpu().numpy()
            else:
                embedding = rna_embedding.data.cpu().numpy()
            if self.args.train_mode == 'test_print':
                test_num, embedding_size = embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
            if self.args.rnaoratac == 'atac':
                embedding = atac_cell_prediction.data.cpu().numpy()
            else:
                embedding = rna_cell_prediction.data.cpu().numpy()
            if self.args.train_mode == 'test_print':
                test_num, embedding_size = embedding.shape
                for print_i in range(test_num):
                    fp_pre.write(str(embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_pre.write(' ' + str(embedding[print_i][print_j]))
                    fp_pre.write('\n')

            # calculate accuracy, ignore testinf data with unknown labels(-1)
            pred = atac_cell_prediction.data.max(1)[1]
            atac_cell_correct += pred[cell_target >= 0].eq(cell_target[cell_target >= 0].data).cpu().sum()
            atac_cell_total += cell_target[cell_target >= 0].size(0)
            err = 100. - 100. * atac_cell_correct.float() / atac_cell_total
            progress_bar(batch_idx, len(self.test_loader),
                         'ATAC CELL Err: %.3f%% (%d/%d)' % (err, atac_cell_total - atac_cell_correct, atac_cell_total))

        if self.args.train_mode == 'test_print':
            fp_em.close()
            fp_pre.close()

        # save checkpoint
        if err < self.best_pred:
            self.best_pred = err
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'model_cell_state_dict': self.model_cell.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_cell.state_dict(),
            'best_pred': self.best_pred,
        }, args=self.args, is_best=is_best)
