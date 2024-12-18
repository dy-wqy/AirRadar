import time
import numpy as np
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
import torch.nn.functional as F

from src.base.trainer import BaseTrainer

from src.utils import graph_algo

from functools import partial


from tensorboardX import SummaryWriter


class Trainer(BaseTrainer):
    def __init__(self, **args):
        super(Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr)
        self._supports = self._calculate_supports(args["adj_mat"], args["filter_type"])
        self._lr_scheduler = MultiStepLR(
            self._optimizer, self._steps, gamma=self._lr_decay_ratio
        )
        # self.alpha = 2
        self.alpha = 0.1
        self.beta = 1
        # self.alpha = 0.00001
        print("alpha:", self.alpha)
        self.rec_mae = nn.L1Loss()
        self.rec_mse = nn.MSELoss()
    def _calculate_supports(self, adj_mat, filter_type):
        # For GNNs, not for AirFormer
        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "scalap":
            supports = [graph_algo.calculate_scaled_laplacian(new_adj).todense()]
        elif filter_type == "normlap":
            supports = [
                graph_algo.calculate_normalized_laplacian(new_adj)
                .astype(np.float32)
                .todense()
            ]
        elif filter_type == "symnadj":
            supports = [graph_algo.sym_adj(new_adj)]
        elif filter_type == "transition":
            supports = [graph_algo.asym_adj(new_adj)]
        elif filter_type == "doubletransition":
            supports = [
                graph_algo.asym_adj(new_adj),
                graph_algo.asym_adj(np.transpose(new_adj)),
            ]
        elif filter_type == "identity":
            supports = [np.diag(np.ones(new_adj.shape[0])).astype(np.float32)]
        elif filter_type == "dglgraph":
            supports = graph_algo.dgl_adj(new_adj)
        else:
            error = 0
            assert error, "adj type not defined"
        # supports = [torch.tensor(i).cuda() for i in supports]
        # supports = None
        return supports

    def train_batch(self, X, label, History, pred_attr, iter):
        """
        the training process of a batch
        """
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred, mask = self.model(X, History, None, pred_attr, g=supports)
        if pred_attr == "PM25":
            label = label[..., 0].unsqueeze(-1)  # PM2.5
        pred = self._inverse_transform([pred])[0]

        mask = 1 - mask
        pred_mask = pred[mask.nonzero(as_tuple=True)]
        label_mask = label[mask.nonzero(as_tuple=True)]

        non_missing = label_mask > 1
        pred_mask = pred_mask[non_missing]
        label_mask = label_mask[non_missing]

        # mae_loss = self.sce_loss(pred_mask, label_mask)
        mae_loss = self.rec_mae(pred_mask, label_mask)

        loss = mae_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self._clip_grad_value
        )
        self.optimizer.step()
        return loss.item(), mae_loss.item()

    def train(self, pred_attr="PM25"):
        """
        rewrite the train process due to the stochastic stage
        """
        self.logger.info("start training !!!!!")
        write_path = self._save_path + "/loss_logs/"
        if os.path.exists(write_path):
            writer = SummaryWriter(write_path)
        else:
            os.makedirs(write_path)
            writer = SummaryWriter(write_path)
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()

            train_losses = []
            mae_losses = []
            recon_losses = []
            kl_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break
            start_time = time.time()
            for i, (X, label, History) in enumerate(self.data["train_loader"]):
                X, label, history = self._check_device([X, label, History])
                loss, mae_loss = self.train_batch(
                    X, label, history, pred_attr, iter
                )
                train_losses.append(loss)
                mae_losses.append(mae_loss)
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate(pred_attr)
                        message = "Epoch [{}/{}] ({}) train_mae: {:.4f}, \
                            val_mae: {:.4f} ".format(
                            epoch,
                            self._max_epochs,
                            iter,
                            np.mean(mae_losses),
                            val_loss,
                        )
                        writer.add_scalars(
                            "Loss/",
                            {
                                "train mae": float(np.mean(train_losses)),
                                "validation mae": float(val_loss),
                            },
                            iter,
                        )
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp
                            )
                            self._logger.info(
                                "Val loss decrease from {:.4f} to {:.4f}, "
                                "saving to {}".format(
                                    np.min(val_losses), val_loss, model_file_name
                                )
                            )

                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # val_loss =0
            val_loss = self.evaluate(pred_attr)

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_lr()[0]

            message = (
                "Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, "
                "{:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    np.mean(train_losses),
                    val_loss,
                    new_lr,
                    (end_time - start_time),
                )
            )
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}, "
                    "saving to {}".format(np.min(val_losses), val_loss, model_file_name)
                )
                val_losses.append(val_loss)
                saved_epoch = epoch

    def test_batch(self, X, label, history, mask_nodes, pred_attr):
        if pred_attr == "PM25":
            label = label[..., 0].unsqueeze(-1)
        pred, mask  = self.model(X, history, mask_nodes, pred_attr, self.supports)
        mask = 1 - mask

        pred = self._inverse_transform([pred])[0]
        return pred, label, mask
