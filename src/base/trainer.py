import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam

from src.utils.logging import get_logger
from src.utils import metrics as mc
from src.utils.metrics import masked_mae

from src.base.sampler import RandomSampler

import pandas as pd


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        adj_mat,
        filter_type: str,
        data,
        aug: float,
        base_lr: float,
        steps,
        lr_decay_ratio,
        log_dir: str,
        n_exp: int,
        save_iter: int = 300,
        clip_grad_value: Optional[float] = None,
        max_epochs: Optional[int] = 1000,
        patience: Optional[int] = 1000,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, "info_{}.log".format(n_exp), level=logging.INFO
        )
        if device is None:
            print(
                "`device` is missing, try to train and evaluate the model on default device."
            )
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model
        self.model.to(self._device)
        self._logger.info(
            "the number of parameters: {}".format(self.model.param_num(self.model.name))
        )

        self._adj_mat = adj_mat
        self._filter_type = filter_type
        self._aug = aug
        self._loss_fn = masked_mae
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)
        self._lr_decay_ratio = lr_decay_ratio
        self._steps = steps
        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(
                self.optimizer, steps, gamma=lr_decay_ratio
            )
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._data = data[0]
        self._mask_nodes = data[1]
        self._supports = None

        if aug > 0:
            self._sampler = RandomSampler(adj_mat, filter_type)
            # self._sampler = None
        self._supports = self._calculate_supports(adj_mat, filter_type)
        # assert(self._supports is not None)

    @property
    def model(self):
        return self._model

    @property
    def supports(self):
        return self._supports

    @property
    def data(self):
        return self._data

    @property
    def mask_nodes(self):
        return self._mask_nodes

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        n_output_dim = 1
        target_dim = 0
        # 0 for PM25, 1 for PM10, 2 for NO2, 3 for CO, 4 for O3, 5 for SO2

        def inv(tensor, scalers):
            for i in range(n_output_dim):
                tensor[..., i] = scalers[target_dim].inverse_transform(tensor[..., i])
            return tensor

        if isinstance(tensors, list):
            return [inv(tensor, self.data["scalers"]) for tensor in tensors]
        else:
            return inv(tensors, self.data["scalers"])

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = "final_model_{}.pt".format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return True

    def load_model(self, epoch, save_path, n_exp):
        filename = "final_model_{}.pt".format(n_exp)
        self.model.load_state_dict(torch.load(os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info(
            "Early stop at epoch {}, loss = {:.6f}".format(epoch, best_loss)
        )
        np.savetxt(
            os.path.join(self.save_path, "val_loss_{}.txt".format(self._n_exp)),
            [best_loss],
            fmt="%.4f",
            delimiter=",",
        )

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

    def train_batch(self, X, label, pred_attr, iter):
        """
        the training process of a batch
        """
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred, _ = self.model(X, pred_attr)
        if pred_attr == "PM10":
            label = label[..., 1].unsqueeze(-1)  # PM10
        elif pred_attr == "PM25":
            label = label[..., 0].unsqueeze(-1)  # PM2.5
        elif pred_attr == "NO2":
            label = label[..., 2].unsqueeze(-1)  # NO2
        elif pred_attr == "CO":
            label = label[..., 3].unsqueeze(-1)  # CO
        elif pred_attr == "O3":
            label = label[..., 4].unsqueeze(-1)  # O3
        elif pred_attr == "SO2":
            label = label[..., 5].unsqueeze(-1)  # SO2
        pred = self._inverse_transform([pred])[0]
        mask = 1 - mask
        mask_index = mask.nonzero(as_tuple=True)
        pred = pred[mask_index]
        label = label[mask_index]

        non_missing = label > 0
        pred = pred[non_missing]
        label = label[non_missing]

        loss = self.loss_fn(pred, label, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self._clip_grad_value
        )
        self.optimizer.step()
        return loss.item()

    def train(self, pred_attr="PM25"):
        """
        the training process
        """
        self.logger.info("start training !!!!!")
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, (X, label) in enumerate(self.data["train_loader"]):
                X, label = self._check_device([X, label])
                train_losses.append(self.train_batch(X, label, pred_attr, iter))
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:  # iteration needs to be checked
                        val_loss = self.evaluate(pred_attr)
                        message = "Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} ".format(
                            epoch,
                            self._max_epochs,
                            iter,
                            np.mean(train_losses),
                            val_loss,
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

            if val_loss < np.min(val_losses):  # error saving criterion
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}, "
                    "saving to {}".format(np.min(val_losses), val_loss, model_file_name)
                )
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self, pred_attr):
        """
        model evaluation
        """
        labels = []
        preds = []
        mask_nodes = self.mask_nodes["val"]
        masks = []
        with torch.no_grad():
            self.model.eval()
            for _, (X, label, History) in enumerate(self.data["val_loader"]):
                X, label, history = self._check_device([X, label, History])
                pred, label, mask = self.test_batch(
                    X, label, history, mask_nodes, pred_attr
                )
                labels.append(label.cpu())
                preds.append(pred.cpu())
                masks.append(mask.cpu())

        masks = torch.cat(masks, dim=0).unsqueeze(-1)
        valid_count = torch.sum(masks)

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mask_index = masks.nonzero(as_tuple=True)
        preds = preds[mask_index]
        labels = labels[mask_index]

        non_missing = labels > 0
        preds = preds[non_missing]
        labels = labels[non_missing]

        mae = torch.abs((preds - labels)).sum() / valid_count
        return mae

    def test_batch(self, X, label, history, mask_nodes, pred_attr):
        """
        the test process of a batch
        """
        if pred_attr == "PM25":
            label = label[..., 0].unsqueeze(-1)  # PM2.5
        elif pred_attr == "PM10":
            label = label[..., 1].unsqueeze(-1)  # PM10
        elif pred_attr == "NO2":
            label = label[..., 2].unsqueeze(-1)  # NO2
        elif pred_attr == "CO":
            label = label[..., 3].unsqueeze(-1)  # CO
        elif pred_attr == "O3":
            label = label[..., 4].unsqueeze(-1)  # O3
        elif pred_attr == "SO2":
            label = label[..., 5].unsqueeze(-1)  # SO2
        # label = X[..., 0].unsqueeze(-1)  # PM2.5
        pred, mask, _ = self.model(X, history, mask_nodes, pred_attr, self.supports)
        mask = 1 - mask
        # pred = pred[mask.nonzero(as_tuple=True)]
        # label = label[mask.nonzero(as_tuple=True)]
        pred = self._inverse_transform([pred])[0]
        return pred, label

    def test(self, epoch, mode="test", pred_attr="PM25"):
        """
        test process
        """
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []
        masks = []
        mask_nodes = self.mask_nodes["test"]
        with torch.no_grad():
            self.model.eval()
            for _, (X, label, History) in enumerate(self.data[mode + "_loader"]):
                X, label, History = self._check_device([X, label, History])
                pred, label, mask = self.test_batch(
                    X, label, History, mask_nodes, pred_attr
                )

                labels.append(label.cpu())
                preds.append(pred.cpu())
                masks.append(mask.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        masks = torch.cat(masks, dim=0).unsqueeze(-1)
        mask_index = masks.nonzero(as_tuple=True)
        preds = preds[mask_index]
        labels = labels[mask_index]

        non_missing = labels > 0

        preds = preds[non_missing]
        labels = labels[non_missing]
        valid_count = preds.shape[0]

        sorted_label, _ = torch.sort(labels)

        mae = torch.abs((preds - labels)).sum() / valid_count
        rmse = torch.sqrt((((preds - labels) ** 2)).sum() / (valid_count + 1e-7))
        mape = torch.abs((preds - labels) / (labels + 1e-6)).sum() / valid_count
        # mae = self.loss_fn(preds, labels, 0.0).item()
        # mse = F.mse_loss(preds, labels)
        # rmse = torch.sqrt(mse)
        # mape = torch.mean(torch.abs(labels - preds) / labels)

        print(
            " test_mae: {:.4f}, test_rmse: {:.4f}, test_mape: {:.4f} ".format(
                mae, rmse, mape
            )
        )
        # if self.model.horizon == 24:
        #     amae_day = []
        #     armse_day = []

        #     for i in range(0, self.model.horizon, 8):
        #         pred = preds[:, i: i + 8]
        #         real = labels[:, i: i + 8]
        #         metrics = mc.compute_all_metrics(pred, real, 0.0)
        #         amae_day.append(metrics[0])
        #         armse_day.append(metrics[1])

        #     log = '0-7 (1-24h) Test MAE: {:.4f}, Test RMSE: {:.4f}'
        #     print(log.format(amae_day[0], armse_day[0]))
        #     log = '8-15 (25-48h) Test MAE: {:.4f},  Test RMSE: {:.4f}'
        #     print(log.format(amae_day[1], armse_day[1]))
        #     log = '16-23 (49-72h) Test MAE: {:.4f},  Test RMSE: {:.4f}'
        #     print(log.format(amae_day[2], armse_day[2]))

        #     results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(4))
        #     Time_list=['1-24h','25-48h','49-72h', 'SuddenChange']
        #     for i in range(3):
        #         results.iloc[i, 0]= Time_list[i]
        #         results.iloc[i, 1]= amae_day[i]
        #         results.iloc[i, 2]= armse_day[i]

        # else:
        #     print('The output length is not 24!!!')

        # mask_sudden_change = mc.sudden_changes_mask(labels, datapath = './data/AIR_TINY', null_val = 0.0, threshold_start = 75, threshold_change = 20)
        # results.iloc[3, 0] = Time_list[3]
        # sc_mae, sc_rmse = mc.compute_sudden_change(mask_sudden_change, preds, labels, null_value = 0.0)
        # results.iloc[3, 1:] = [sc_mae, sc_rmse]
        # log = 'Sudden Changes MAE: {:.4f},  RMSE: {:.4f}'
        # print(log.format(sc_mae, sc_rmse))
        # results.to_csv(os.path.join(self.save_path, 'metrics_{}.csv'.format(self._n_exp)), index = False)

    def save_preds(self, epoch):
        """
        save prediction results
        """
        self.load_model(epoch, self.save_path, self._n_exp)

        for mode in ["train", "val", "test"]:
            labels = []
            preds = []
            inputs = []
            with torch.no_grad():
                self.model.eval()
                for _, (X, label) in enumerate(self.data[mode + "_loader"]):
                    X, label = self._check_device([X, label])
                    pred, label = self.test_batch(X, label)
                    labels.append(label.cpu())
                    preds.append(pred.cpu())
                    inputs.append(X.cpu())
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            inputs = torch.cat(inputs, dim=0)

            np.save(os.path.join(self.save_path, mode + "_preds.npy"), preds)
            np.save(os.path.join(self.save_path, mode + "_labels.npy"), labels)
