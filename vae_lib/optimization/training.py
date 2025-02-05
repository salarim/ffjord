from __future__ import print_function
import time
import torch
import os
import matplotlib.pyplot as plt

from vae_lib.optimization.loss import calculate_loss
from vae_lib.utils.visual_evaluation import plot_reconstructions, plot_images
from vae_lib.utils.log_likelihood import calculate_likelihood
from vae_lib.utils.load_data import visualize_synthetic_data

import numpy as np
from train_misc import count_nfe, override_divergence_fn


def train(epoch, train_loader, model, opt, args, logger):

    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    # set warmup coefficient
    beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
    logger.info('beta = {:5.4f}'.format(beta))
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        if args.dynamic_binarization:
            data = torch.bernoulli(data)

        data = data.view(-1, *args.input_size)

        opt.zero_grad()
        
        if args.conditional:
            x_mean, z_mu, z_var, ldj, z0, zk = model(data, target)
        else:
            x_mean, z_mu, z_var, ldj, z0, zk = model(data)
        
        # if batch_idx == len(train_loader)-1:
        #     print('-'*10 ,)
        # for i in range(len(x_mean)):
        #     print(x_mean[i].data[0].item(), x_mean[i].data[1].item(), data[i].data[0].item(), data[i].data[1].item())
        if 'cnf' in args.flow:
            f_nfe = count_nfe(model)

        loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        loss.backward()

        if 'cnf' in args.flow:
            t_nfe = count_nfe(model)
            b_nfe = t_nfe - f_nfe

        train_loss[batch_idx] = loss.item()
        train_bpd[batch_idx] = bpd

        opt.step()

        rec = rec.item()
        kl = kl.item()

        num_data += len(data)

        batch_time = time.time() - end
        end = time.time()

        if batch_idx % args.log_interval == 0:
            if args.input_type == 'binary':
                perc = 100. * batch_idx / len(train_loader)
                log_msg = (
                    'Epoch {:3d} [{:5d}/{:5d} ({:2.0f}%)] | Time {:.3f} | Loss {:11.6f} | '
                    'Rec {:11.6f} | KL {:11.6f}'.format(
                        epoch, num_data, len(train_loader.sampler), perc, batch_time, loss.item(), rec, kl
                    )
                )
            else:
                perc = 100. * batch_idx / len(train_loader)
                tmp = 'Epoch {:3d} [{:5d}/{:5d} ({:2.0f}%)] | Time {:.3f} | Loss {:11.6f} | Bits/dim {:8.6f}'
                log_msg = tmp.format(epoch, num_data, len(train_loader.sampler), perc, batch_time, loss.item(),
                                     bpd), '\trec: {:11.3f}\tkl: {:11.6f}\tvar: {}'.format(rec, kl, torch.mean(torch.mean(z_var, dim=0)))
                log_msg = "".join(log_msg)
            if 'cnf' in args.flow:
                log_msg += ' | NFE Forward {} | NFE Backward {}'.format(f_nfe, b_nfe)
            logger.info(log_msg)

    if args.input_type == 'binary':
        logger.info('====> Epoch: {:3d} Average train loss: {:.4f}'.format(epoch, train_loss.sum() / len(train_loader)))
    else:
        logger.info(
            '====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.
            format(epoch, train_loss.sum() / len(train_loader), train_bpd.sum() / len(train_loader))
        )

    return train_loss


def evaluate(data_loader, model, args, logger, testing=False, epoch=0):
    model.eval()
    loss = 0.
    batch_idx = 0
    bpd = 0.

    if args.input_type == 'binary':
        loss_type = 'elbo'
    else:
        loss_type = 'bpd'

    if testing and 'cnf' in args.flow:
        override_divergence_fn(model, "brute_force")

    for data, target in data_loader:
        batch_idx += 1

        with torch.no_grad():
            # if args.cuda:
            #     data.to('cuda')
            #     target.to('cuda')
            data = data.view(-1, *args.input_size)
            
            if args.conditional:
                x_mean, z_mu, z_var, ldj, z0, zk = model(data.to('cuda'), target.to('cuda'))
            else:
                x_mean, z_mu, z_var, ldj, z0, zk = model(data.to('cuda'))

            batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data.to('cuda'), z_mu, z_var, z0, zk, ldj, args)

            bpd += batch_bpd
            loss += batch_loss.item()

            # PRINT RECONSTRUCTIONS
            if batch_idx == 1 and testing is False:
                if args.input_type == 'synthetic':
                    sample_size = 500
                    normal_sample = torch.FloatTensor(sample_size* args.num_labels * args.z_size).normal_().reshape(sample_size*args.num_labels,-1).to(args.device)
                    if args.conditional:
                        tgt = torch.tensor(list(range(args.num_labels))*sample_size).to(args.device)
                        sample = model.decode(normal_sample, tgt)
                    else:
                        sample = model.decode(normal_sample, None)
                    visualize_synthetic_data(sample.cpu().numpy(), tgt.cpu().numpy(), args.num_labels, 'rec')
                elif not args.evaluate:
                    plot_reconstructions(data, x_mean, batch_loss, loss_type, epoch, args)
                    sample_lables_num = args.num_labels - 1
                    normal_sample = torch.FloatTensor(sample_lables_num * args.z_size).normal_().reshape(sample_lables_num,-1).to(args.device)
                    if args.conditional:
                        tgt = torch.tensor(list(range(sample_lables_num))).to(args.device)
                        sample = model.decode(normal_sample, tgt)
                    else:
                        sample = model.decode(normal_sample, None)
                    plot_images(args, sample.data.cpu().numpy(), args.snap_dir + 'reconstruction/', 'sample_of_1_e_'+str(epoch))
                else:
                    print('###############################')
                    sample_size = 100
                    normal_sample = torch.FloatTensor(sample_size * args.num_labels * args.z_size).normal_().reshape(sample_size * args.num_labels,-1).to(args.device)
                    if args.conditional:
                        sample_labels = []
                        for i in range(args.num_labels):
                            for j in range(sample_size):
                                sample_labels.append(i)
                        tgt = torch.tensor(sample_labels).to(args.device)
                        import cv2
                        samples = model.decode(normal_sample, tgt)
                        samples, tgt = samples.data.cpu().numpy(), tgt.data.cpu().numpy()
                        if not os.path.exists(args.snap_dir + 'samples/'):
                            os.makedirs(args.snap_dir + 'samples/')
                        for i, sample in enumerate(samples):
                            l = tgt[i]
                            if not os.path.exists(args.snap_dir + 'samples/' + str(l)):
                                os.makedirs(args.snap_dir + 'samples/' + str(l))
                            sample = sample.swapaxes(0, 2)
                            sample = sample.swapaxes(0, 1)
                            sample = sample * 255
                            sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(args.snap_dir + 'samples/' + str(l) + '/' + str(i) + '.jpg', sample)

    loss /= len(data_loader)
    bpd /= len(data_loader)

    if testing:
        logger.info('====> Test set loss: {:.4f}'.format(loss))

    # Compute log-likelihood
    if testing and not ("cnf" in args.flow):  # don't compute log-likelihood for cnf models

        with torch.no_grad():
            test_data = data_loader.dataset.tensors[0]

            if args.cuda:
                test_data = test_data.cuda()

            logger.info('Computing log-likelihood on test set')

            model.eval()

            if args.dataset == 'caltech':
                log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, logger, S=2000, MB=500)
            else:
                log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, logger, S=5000, MB=500)

        if 'cnf' in args.flow:
            override_divergence_fn(model, args.divergence_fn)
    else:
        log_likelihood = None
        nll_bpd = None

    if args.input_type in ['multinomial']:
        bpd = loss / (np.prod(args.input_size) * np.log(2.))

    if testing and not ("cnf" in args.flow):
        logger.info('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

        if args.input_type != 'binary':
            logger.info('====> Test set bpd (elbo): {:.4f}'.format(bpd))
            logger.info(
                '====> Test set bpd (log-likelihood): {:.4f}'.
                format(log_likelihood / (np.prod(args.input_size) * np.log(2.)))
            )

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd
