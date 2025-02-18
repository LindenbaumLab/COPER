from itertools import chain
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from pytorch_lightning import Trainer, seed_everything
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
from sklearn.utils import shuffle


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    print(cfg)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(cfg.seed)
    np.random.seed(cfg.seed)
    if not os.path.exists(cfg.logs_dir): os.makedirs(cfg.logs_dir)
    model = CoperModel(cfg)
    logger = TensorBoardLogger(
        save_dir=cfg.logs_dir,
        name=f"{os.path.basename(__file__)}_{os.path.basename(args.cfg)}_",
        log_graph=False)
    trainer = Trainer(**cfg.trainer, callbacks=[LearningRateMonitor(logging_interval='step'), ])
    trainer.logger = logger
    trainer.fit(model)
    print(f"ACC: {model.best_accuracy}")
    print(f"ARI: {model.best_ari}")
    print(f"NMI: {model.best_nmi}")


class CoperModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # dataset initialization
        self.dataset = MultiviewDataset(cfg)
        print(f"Dataset length: {self.dataset.__len__()}")
        if self.dataset.__len__() < self.cfg.batch_size:
            self.cfg.batch_size = self.dataset.__len__()
        for view_id in self.dataset.views.keys():
            print(f"X.min()={self.dataset.views[view_id].min()}, X.max()={self.dataset.views[view_id].max()}")
        self.cfg.input_dim = self.dataset.num_features()
        self.cfg.n_clusters = self.dataset.num_classes()
        self.cfg.num_views = len(self.dataset.views)
        for y in range(self.dataset.num_classes()):
            print(f"Label {y}: {len(self.dataset.labels['view1'][self.dataset.labels['view1'] == y])}")
        num_views = len(self.dataset.views)

        # module helpers
        self.save_hyperparameters()
        self.best_evaluation_stats = {}
        self.best_accuracy = {f'common': -1000}
        self.best_ari = {'common': -1000}
        self.best_nmi = {'common': -1000}
        self.max_silhouette_score = []
        self.min_dbi_score = []

        # neural networks
        self.encdec = EncoderDecoder(cfg, num_views)
        self.cluster_model = ClusteringLayer(cfg, num_views)

        # cca loss initialization
        self.cca_loss = CCALoss()

        # view pairs for permutations
        views_index = list(range(self.cfg.num_views))
        self.view_pairs = [(views_index[i], views_index[j])
                           for i in range(len(views_index))
                           for j in
                           range(i + 1, len(views_index))]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=False, shuffle=False, num_workers=0)


    def training_step(self, batch, batch_idx):
        loss = 0
        for view_idx1, view_idx2 in self.view_pairs:
            if self.cfg.get("use_decoder", True):
                loss = loss + self.reconstruction_loss(batch[view_idx1 * 2], f'view{view_idx1 + 1}')
                loss = loss + self.reconstruction_loss(batch[view_idx2 * 2], f'view{view_idx2 + 1}')
            try:
                cca_loss = self.calc_cca_loss(batch, view_idx1, view_idx2)
                self.log(f"train/cca_loss_real", cca_loss.item(), sync_dist=True)
                loss = loss + cca_loss / abs(cca_loss.item()) / len(self.view_pairs)
            except:
                pass
        if self.current_epoch >= self.cfg.stage_II_epochs:
            ce_loss, pseudo_batch = self.pseudo_labels_cross_entropy_loss(batch)
            loss = loss + ce_loss  # / abs(ce_loss.item())

        if self.current_epoch >= self.cfg.stage_III_epochs and self.cfg.train_with_pseudo_labels_shuffle:
            pseudo_batch = self.generate_corresponding_batch_from_pseudo_labels(pseudo_batch)
            pseudo_batch = [item.to(batch[0].device) for item in pseudo_batch]

            if self.cfg.get('pseudo_cca_loss', False):
                for view_idx1, view_idx2 in self.view_pairs:
                    try:
                        cca_loss_pseudo_permut = self.calc_cca_loss(pseudo_batch, view_idx1, view_idx2)
                        self.log(f"train/cca_loss_pseudo_permut", cca_loss_pseudo_permut.item(), sync_dist=True)
                        loss = loss + self.cfg.get('pseudo_cca_loss_lambda', 0.01) * cca_loss_pseudo_permut / abs(
                            cca_loss_pseudo_permut.item()) / len(self.view_pairs)
                    except:
                        pass
            x_batch = [pseudo_batch[i * 2] for i in range(self.cfg.num_views)]
            y_batch = [pseudo_batch[i * 2 + 1] for i in range(self.cfg.num_views)]
            ps_ce_loss_all_views = 0
            for i in range(self.cfg.num_views):
                ps_ce_loss_all_views += self.pseudo_labels_cross_entropy_loss_per_view(x_batch, y_batch, i)
            loss = loss + ps_ce_loss_all_views
        if isinstance(loss, int):
            self.log("train/loss", loss)
        else:
            self.log("train/loss", loss.item())
        return loss


    def validation_step(self, batch, batch_idx):
        e_views = []
        for i in range(self.cfg.num_views):
            e_views.append(self.encdec.encoders[f'view{i + 1}'](batch[i * 2]))
        y_hat_views = self.cluster_model(e_views)
        y_hat_common = y_hat_views.argmax(dim=-1)  # [ 1, Batch, Clusters]
        y_common = batch[1]  # could be any view
        self.val_dict['common']["val_label_list"].append(y_common.cpu())
        self.val_dict['common']["val_cluster_list"].append(y_hat_common.cpu())
        self.val_dict['common']["val_emb_list"].append(self.cluster_model.fusion(e_views))

    def on_validation_epoch_start(self):
        self.val_dict = {
            "common": {
                "val_cluster_list": [],
                "val_label_list": [],
                "val_emb_list": []
            },
        }

    def on_validation_epoch_end(self, view="common"):
        cluster_mtx = torch.cat(self.val_dict[view]['val_cluster_list'], dim=0)
        label_mtx = torch.cat(self.val_dict[view]['val_label_list'], dim=0)
        acc_single = self.clustering_accuracy(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1)
        NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        if self.best_accuracy[view] <= acc_single:
            print(f"New best accuracy {view}:", acc_single)
            self.best_accuracy[view] = acc_single
            self.best_ari[view] = ARI
            self.best_nmi[view] = NMI
        self.log(f'val/acc_single_{view}', acc_single, sync_dist=True)  # this is ACC
        self.log(f'val/NMI_{view}', NMI, sync_dist=True)
        self.log(f'val/ARI_{view}', ARI, sync_dist=True)
        try:
            silhouette_score_embs = silhouette_score(
                torch.cat(self.val_dict['common']["val_emb_list"], dim=0).cpu().numpy(),
                cluster_mtx.numpy())
            self.log(f'val/silhouette_score_embs', silhouette_score_embs)
            self.max_silhouette_score.append(silhouette_score_embs)
            dbi_score = davies_bouldin_score(
                torch.cat(self.val_dict['common']["val_emb_list"], dim=0).cpu().numpy(),
                cluster_mtx.numpy())
            self.log(f'val/dbi_score_embs', dbi_score)
            self.min_dbi_score.append(dbi_score)
        except:
            pass

    def configure_optimizers(self):
        cluster_optimizer = torch.optim.Adam(
            chain(
                self.cluster_model.model.parameters(),
                self.cluster_model.fusion.parameters(),
                self.encdec.parameters(),
            ),
            lr=self.cfg.get('lr', 1e-4))
        return cluster_optimizer

    #=========================== loss methods:========================================================
    def pseudo_labels_cross_entropy_loss_per_view(self, x_hat, y_hat, view_idx):
        e = [self.encdec.encoders[f'view{view_idx + 1}'](x_hat[view_idx])]
        c = self.cluster_model(e, self.current_epoch)
        loss = F.cross_entropy(c, y_hat[view_idx])
        self.log(f"train/ce_view_{view_idx + 1}_loss", loss.item(), sync_dist=True)
        return loss

    def pseudo_labels_cross_entropy_loss(self, batch):
        pseudo_batch = self.get_pseudo_labels(batch)
        loss = 0
        x_batch = [pseudo_batch[i * 2] for i in range(self.cfg.num_views)]
        y_batch = [pseudo_batch[i * 2 + 1] for i in range(self.cfg.num_views)]
        for i in range(self.cfg.num_views):
            loss = loss + self.pseudo_labels_cross_entropy_loss_per_view(x_batch, y_batch, i)
        return loss, pseudo_batch

    def reconstruction_loss(self, x, view):
        x_hat = self.encdec(x, view)
        loss = F.mse_loss(x_hat, x)
        self.log(f'train/encdec_{view}_loss', loss.item(), sync_dist=True)
        return loss

    def calc_cca_loss(self, batch, view_idx1, view_idx2):
        e_view1 = self.encdec.encoders[f'view{view_idx1 + 1}'](batch[view_idx1 * 2])
        e_view2 = self.encdec.encoders[f'view{view_idx2 + 1}'](batch[view_idx2 * 2])
        cca_loss = self.cca_loss(e_view1, e_view2)
        return cca_loss

    # =========================== evaluation metrics ========================================================
    @staticmethod
    def clustering_accuracy(cluster_mtx, label_mtx, n_classes=10):
        cluster_indx = list(cluster_mtx.unique())
        assigned_label_list = []
        assigned_count = []
        while (len(assigned_label_list) <= n_classes) and len(cluster_indx) > 0:
            max_label_list = []
            max_count_list = []
            for indx in cluster_indx:
                # calculate highest number of matchs
                mask = cluster_mtx == indx
                label_elements, counts = label_mtx[mask].unique(return_counts=True)
                for assigned_label in assigned_label_list:
                    counts[label_elements == assigned_label] = 0
                max_count_list.append(counts.max())
                max_label_list.append(label_elements[counts.argmax()])
            max_label = torch.stack(max_label_list)
            max_count = torch.stack(max_count_list)
            assigned_label_list.append(max_label[max_count.argmax()])
            assigned_count.append(max_count.max())
            cluster_indx.pop(max_count.argmax().item())
        total_correct = torch.tensor(assigned_count).sum().item()
        total_sample = cluster_mtx.shape[0]
        acc = total_correct / total_sample
        return acc

    # =========================== COPER methods ========================================================
    @torch.no_grad()
    def get_pseudo_labels_for_view(self, x, view_id, top_logits_idx, e):
        # pseudo labels
        k = self.cfg.pseudo_labels_k
        reliable_labels_matrix = - torch.ones(x.size(0), self.cfg.n_clusters, device=x.device)
        reliable_labels_probs = - torch.zeros(x.size(0), self.cfg.n_clusters, device=x.device).float()
        y_hat = []  # [N]
        x_hat = []
        idx = []
        min_cosine_vals = torch.zeros(self.cfg.n_clusters)
        max_cosine_vals = torch.zeros(self.cfg.n_clusters)

        # reliable labels
        for i in range(self.cfg.n_clusters):
            centroid = e[top_logits_idx[:, i]].mean(dim=0).reshape(1, -1)
            cosine_sim = torch.cosine_similarity(centroid, e)
            min_cosine_vals[i] = cosine_sim.min()
            max_cosine_vals[i] = cosine_sim.max()
            cluster_samples_idx = torch.topk(cosine_sim, k=k, dim=0).indices
            if cosine_sim[cluster_samples_idx].max() > self.cfg.cosine_neighbor_threshold:
                cluster_samples_idx = np.array([idx.item()
                                                for idx in cluster_samples_idx
                                                if (cosine_sim[idx] > self.cfg.cosine_neighbor_threshold)
                                                ])

            reliable_labels_matrix[:, i][cluster_samples_idx] = i
            reliable_labels_probs[:, i][cluster_samples_idx] = 0.5 * (1 + cosine_sim[cluster_samples_idx])

        for i in range(x.size(0)):
            sample_labels = reliable_labels_matrix[i].clone()
            sample_label_probs = reliable_labels_probs[i].clone()

            if len(sample_labels[sample_labels > -1]) > 1:
                probs = sample_label_probs / sample_label_probs.sum()
                y_hat.append(probs.reshape(1, -1))
                x_hat.append(x[i].reshape(1, -1))
                idx.append(i)
            elif len(sample_labels[
                         sample_labels > -1]) == 1:
                y_hat.append(F.one_hot(sample_labels[sample_labels > -1].long(),
                                       num_classes=self.cfg.n_clusters).float().reshape(1, -1))
                x_hat.append(x[i].reshape(1, -1))
                idx.append(i)

        if len(x_hat) == 0:
            raise ValueError("There is now pseudo labels for the view. Decrease the cosine_neighbor_threshold value.")
        x_hat = torch.cat(x_hat, dim=0)
        y_hat = torch.cat(y_hat, dim=0)
        self.log(f"train/pseudo_labels_view{view_id}", float(x_hat.size(0)), sync_dist=True)
        return x_hat.detach(), y_hat.detach(), idx

    def get_pseudo_labels(self, batch):
        # x_view1, _, x_view2, _ = batch
        x_hat_view_all, y_hat_view_all, idx_view_all = [], [], []
        x_batch = [batch[i * 2] for i in range(self.cfg.num_views)]

        e = []
        for i in range(self.cfg.num_views):
            e.append(self.encdec.encoders[f"view{i + 1}"](x_batch[i]))

        clustering_matrix = self.cluster_model(e)  # [N,K]
        result_batch = []

        clustering_matrix = torch.softmax(clustering_matrix, dim=1)
        top_logits_idx = torch.topk(clustering_matrix, k=self.cfg.pseudo_labels_k, dim=0).indices  # [k, K]

        for i in range(self.cfg.num_views):
            x_hat_view, y_hat_view, idx_view = self.get_pseudo_labels_for_view(x_batch[i], i, top_logits_idx, e[i])
            x_hat_view_all.append(x_hat_view)
            y_hat_view_all.append(y_hat_view)
            idx_view_all.append(idx_view)
            result_batch.append(x_hat_view)
            result_batch.append(y_hat_view)

        if self.cfg.apply_reliable_pseudo_labels:
            # for each view we are given a set of samples and with one or more labels.
            # now we want to merge the labels so that:
            # if a sample has a label in some view - we assign this labels to all views for the same sample

            # multi view reliability
            common_idx = set(torch.tensor(idx_view_all[0]).cpu().numpy().tolist())
            for i in range(1, self.cfg.num_views):
                common_idx = common_idx.intersection(torch.tensor(idx_view_all[i]).cpu().numpy().tolist())

            idx_view_all = [torch.tensor(idx_view) for idx_view in idx_view_all]
            result_index_all = [list(range(x_hat_view.size(0))) for x_hat_view in x_hat_view_all]
            for idx in common_idx:
                removed = []
                for i in range(1, self.cfg.num_views):
                    if y_hat_view_all[0][idx_view_all[0] == idx].argmax(-1) != y_hat_view_all[i][idx_view_all[i] == idx].argmax(-1):
                        to_remove = torch.argwhere(idx_view_all[0] == idx).item()
                        if to_remove not in removed:
                            result_index_all[0].remove(to_remove)
                            removed.append(to_remove)
                        result_index_all[i].remove(torch.argwhere(idx_view_all[i] == idx).item())
            result_batch = []
            for i in range(self.cfg.num_views):
                result_batch.append(torch.index_select(x_hat_view_all[i],
                                                       dim=0,
                                                       index=torch.tensor(
                                                           data=result_index_all[i],
                                                           device=x_hat_view_all[i].device).long()))
                result_batch.append(torch.index_select(y_hat_view_all[i],
                                                       dim=0,
                                                       index=torch.tensor(
                                                           data=result_index_all[i],
                                                           device=x_hat_view_all[i].device).long()))
        return result_batch

    def generate_corresponding_batch_from_pseudo_labels(self, batch):
        """ batch cstructure: x1,y1,x2,y2,x3,y3,... """
        for i in range(len(batch)):
            if i % 2 == 0:
                # x
                batch[i] = batch[i].detach().cpu().numpy()
            else:  # y
                batch[i] = batch[i].argmax(-1).detach().cpu().numpy()
        new_multiview_batch = [[] for _ in batch]
        for y_i in np.unique(batch[1]):
            local_x_view = [batch[view_idx * 2][batch[view_idx * 2 + 1] == y_i].copy()
                            for view_idx in range(self.cfg.num_views)]
            num_samples = [x.shape[0] for x in local_x_view]
            if np.min(num_samples) == 0: continue
            max_samples = min(self.cfg.pseudo_labels_k, np.max(num_samples))
            for i in range(self.cfg.num_views):
                if num_samples[i] < max_samples:
                    # pad the view:
                    pad = max_samples - num_samples[i]
                    local_x_view[i] = np.concatenate([local_x_view[i], np.random.permutation(
                        np.repeat(local_x_view[i], pad // num_samples[i] + 1, axis=0))[:pad]], axis=0)

            for i in range(self.cfg.num_views):
                # x:
                new_multiview_batch[i * 2].append(np.random.permutation(local_x_view[i]))
                new_multiview_batch[i * 2].append(np.random.permutation(local_x_view[i]))
                # y:
                new_multiview_batch[i * 2 + 1].append(y_i.repeat(local_x_view[i].shape[0]))
                new_multiview_batch[i * 2 + 1].append(y_i.repeat(local_x_view[i].shape[0]))

        for i in range(len(batch)):
            if i % 2 == 0:
                new_multiview_batch[i] = torch.tensor(np.concatenate(new_multiview_batch[i], axis=0)).float()
            else:
                new_multiview_batch[i] = torch.tensor(np.concatenate(new_multiview_batch[i], axis=0)).long()

        return new_multiview_batch


class ClusteringLayer(torch.nn.Module):
    def __init__(self, cfg, num_views):
        super().__init__()
        self.cfg = cfg
        self.fusion = WeightedMean(cfg, num_views, [cfg.n_clusters] * num_views)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.clustering_layer[0], self.cfg.clustering_layer[1]),
            torch.nn.BatchNorm1d(self.cfg.clustering_layer[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.cfg.clustering_layer[1], cfg.n_clusters)
        )
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x_all_views, fuse=True, epoch=None):
        if self.training and (epoch is None or epoch < self.cfg.train_fusion_start_epochs):
            with torch.no_grad():
                fused = self.fusion(x_all_views)
        else:
            fused = self.fusion(x_all_views)
        return self.model(fused)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, cfg, num_views):
        super(EncoderDecoder, self).__init__()
        self.cfg = cfg
        self.encoders = []
        self.encoders = torch.nn.ModuleDict(
            {f'view{i + 1}': self.build_view_encoder(i) for i in range(num_views)}
        )
        if self.cfg.get("use_decoder", False):
            self.decoders = torch.nn.ModuleDict({
                f'view{i + 1}': self.build_view_decoder(i) for i in range(num_views)})
        for i in range(num_views):
            self.encoders[f'view{i + 1}'].apply(self.init_weights)
            if self.cfg.get("use_decoder", False):
                self.decoders[f'view{i + 1}'].apply(self.init_weights)

    def build_view_encoder(self, view_id):
        if 'view1' in self.cfg.encdec:
            view_id_str = f'view{view_id + 1}'
            layers = [
                torch.nn.Linear(self.cfg.input_dim[view_id], self.cfg.encdec.get(view_id_str)[0]),
                torch.nn.BatchNorm1d(self.cfg.encdec.get(view_id_str)[0]),
                torch.nn.ReLU()
            ]
            hidden_layers = len(self.cfg.encdec.get(view_id_str)) // 2 + 1
            for layer_idx in range(1, hidden_layers):
                if layer_idx == hidden_layers - 1:
                    layers += [torch.nn.Linear(self.cfg.encdec.get(view_id_str)[layer_idx - 1],
                                               self.cfg.encdec.get(view_id_str)[layer_idx])]
                else:
                    layers += [
                        torch.nn.Linear(self.cfg.encdec.get(view_id_str)[layer_idx - 1],
                                        self.cfg.encdec.get(view_id_str)[layer_idx]),
                        torch.nn.BatchNorm1d(self.cfg.encdec.get(view_id_str)[layer_idx]),
                        torch.nn.ReLU()
                    ]
        else:
            # same encoder for each view
            if len(self.cfg.encdec) == 1:
                # linear encoder:
                layers = [
                    torch.nn.Linear(self.cfg.input_dim[view_id], self.cfg.encdec[0]),
                    torch.nn.BatchNorm1d(self.cfg.encdec[0]),
                ]
            else:
                layers = [
                    torch.nn.Linear(self.cfg.input_dim[view_id], self.cfg.encdec[0]),
                    torch.nn.BatchNorm1d(self.cfg.encdec[0]),
                    torch.nn.ReLU()
                ]
                hidden_layers = len(self.cfg.encdec) // 2 + 1
                for layer_idx in range(1, hidden_layers):
                    if layer_idx == hidden_layers - 1:
                        layers += [torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx])]
                    else:
                        layers += [
                            torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                            torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                            torch.nn.ReLU()
                        ]
        return torch.nn.Sequential(*layers)

    def build_view_decoder(self, view_id):
        hidden_layers = len(self.cfg.encdec) // 2 + 1
        layers = []
        for layer_idx in range(hidden_layers, len(self.cfg.encdec)):
            layers += [
                torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                torch.nn.ReLU()
            ]
        layers += [torch.nn.Linear(self.cfg.encdec[-1], self.cfg.input_dim[view_id])]
        return torch.nn.Sequential(*layers)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x, view):
        if hasattr(self, "decoders"):
            return self.decoders[view](self.encoders[view](x))
        return self.encoders[view](x)


class CCALoss(torch.nn.Module):
    def __init__(self, outdim_size=10, use_all_singular_values=False):
        super(CCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values

    def forward(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-8

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=H1.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            # regularization for more stability :
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0], device=H1.device) * r1))
            U, V = torch.linalg.eigh(trace_TT)
            U = torch.where(U > eps, U, (torch.ones(U.shape, device=H1.device).float() * eps))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class MultiviewDataset(Dataset):
    """
    Creates a multiview dataset object
    cfg.dataset_dir: A dictionary with 6 keys -
        - "dataset_name": The dataset name
        - "dataset_version": The version number of the dataset (in case it is modyfied)
        - "X": A dictionary with the raw data, each key, view_i is a numpy array with of shape (n, p_i)
        - "Y": A numpy array of labels
        - "view_names": A list of corresponding names to each view
        - "sub_sample": A list with two values, the first is a boolean indicating weather to subsample or not. The second is the amound to subsample
    """

    def __init__(self, cfg):
        super().__init__()

        # Loading the dictionary back from the pickle file
        with open(cfg.dataset_path, "rb") as file:
            dataset_dict = pickle.load(file)

        self.views = dataset_dict["X"]
        self.labels = dataset_dict["Y"]
        self.dataset_name = dataset_dict["dataset_name"]

        if dataset_dict["sub_sample"][0] == True:
            self.num_of_sub_samples = dataset_dict["sub_sample"][1]
            self.sub_sample()
        print("Number of views:", len(self.views))
        print("Dimensions:", [v.shape[1] for v in self.views.values()])
        print("Unique labels:", np.unique(list(self.labels.values())[0]))

    def __getitem__(self, index: int):
        out = []
        for i in range(len(self.views.keys())):
            if isinstance(self.views[f"view{i + 1}"][index], np.ndarray):
                out.append(
                    torch.tensor(self.views[f"view{i + 1}"][index].reshape(-1), dtype=torch.float32))
            else:
                out.append(
                    torch.tensor(self.views[f"view{i + 1}"][index].toarray().reshape(-1), dtype=torch.float32))
            out.append(torch.tensor(self.labels[f"view{i + 1}"][index]).long())
        return out

    def __len__(self) -> int:
        return self.views["view1"].shape[0]

    def num_classes(self):
        return np.unique(self.labels["view1"]).shape[0]

    def num_features(self):
        out = [self.views[f"view{i + 1}"].shape[1] for i in range(len(self.views.keys()))]
        return out

    def sub_sample(self):
        chosen_samples, _ = self.pick_samples()
        for view in self.views.keys():
            self.views[view] = self.views[view][chosen_samples,]

    def pick_samples(self):
        unique_labels = np.unique(self.labels)
        num_labels = len(unique_labels)

        samples_per_label = self.num_of_sub_samples // num_labels

        chosen_samples = []
        left_out_samples = []

        for label in unique_labels:
            indices = np.where(self.labels == label)[0]

            if len(indices) < samples_per_label:
                chosen_samples.extend(indices)
            else:
                np.random.shuffle(indices)
                chosen_indices = indices[:samples_per_label]
                left_out_indices = indices[samples_per_label:]
                chosen_samples.extend(chosen_indices)
                left_out_samples.extend(left_out_indices)

        chosen_samples = shuffle(chosen_samples)
        left_out_samples = shuffle(left_out_samples)

        return chosen_samples, left_out_samples


class WeightedMean(torch.nn.Module):
    def __init__(self, cfg, n_views, input_sizes):
        super().__init__()
        self.cfg = cfg
        self.n_views = n_views
        self.weights = torch.nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def get_weighted_sum_output_size(self, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        return [flat_sizes[0]]

    def forward(self, inputs):
        if "normalize_embeddings" not in self.cfg:
            return self._weighted_sum(inputs, self.weights, normalize_weights=True)
        return self._weighted_sum(
            inputs,
            self.weights,
            normalize_weights=True,
            normalize_embeddings=self.cfg.normalize_embeddings)

    @staticmethod
    def _weighted_sum(tensors, weights, normalize_weights=True, normalize_embeddings=True):
        if normalize_weights:
            weights = F.softmax(weights, dim=0)
        if normalize_embeddings:
            tensors_norm = []
            for t in tensors:
                tensors_norm.append(F.normalize(t))
            out = torch.sum(weights[None, None, :] * torch.stack(tensors_norm, dim=-1), dim=-1)
        else:
            out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
        return out


if __name__ == "__main__":
    main()
