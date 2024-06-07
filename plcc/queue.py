import os.path
import numpy as np
import torch
from torch.nn import functional as F

from prettytable import PrettyTable

from tsne import plot_tsne, plot_patches


class TensorQueue(object):
    def __init__(self, k, n, dim):
        self.k = k
        self.n = n

        self.features_queue = torch.zeros((k, n, dim))
        self.queue_ptr = torch.zeros(k)

        self.current_size = torch.zeros(k)

        self.image_inds = torch.zeros(k, n).long()

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels, image_inds=None):
        for i in torch.unique(labels):
            batch_index = torch.where(labels == i)[0]
            update_size = batch_index.shape[0]
            queue_index = [int((x + self.queue_ptr[i]) % self.n) for x in range(update_size)]

            self.queue_ptr[i] = int((update_size + self.queue_ptr[i]) % self.n)
            # self.features_queue[i, queue_index] = features[batch_index]
            self.features_queue[i, queue_index] = F.normalize(features[batch_index], dim=-1, p=2)

            self.current_size[i] = min(self.n, self.current_size[i] + update_size)

            if image_inds is not None:
                self.image_inds[i, queue_index] = image_inds[batch_index]

    def get_features(self):
        return self.features_queue.clone().detach()

    def load_queue(self, features, current_size, image_inds=None):
        self.features_queue = features
        self.current_size = current_size
        if image_inds is not None:
            self.image_inds = image_inds

    def cuda(self, device=None):
        self.features_queue = self.features_queue.cuda(device)
        self.queue_ptr = self.queue_ptr.cuda(device)
        self.current_size = self.current_size.cuda(device)
        self.image_inds = self.image_inds.cuda(device)


class ClusterQueue(object):
    def __init__(self, k, n, dim, t_neg, t_pos, top_k=5, tmp=0.05):
        self.k = k
        self.n = n
        self.dim = dim
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.top_k = int(top_k)
        self.tmp = tmp

        self.cluster_queue = TensorQueue(k=k, n=n, dim=dim)

        # verbose information
        self.label_num = None
        self.corrected_in_num = None
        self.corrected_out_num = None
        self.top_k_prob = None
        self.update_num = None

    @torch.no_grad()
    def init_cluster(self, features, labels, image_inds=None):
        features_all = concat_all_gather(features)
        labels_all = concat_all_gather(labels)
        if image_inds is not None:
            image_inds_all = concat_all_gather(image_inds)
        else:
            image_inds_all = None

        self.cluster_queue.dequeue_and_enqueue(features_all, labels_all, image_inds_all)

    # @torch.no_grad()
    # def calc_distance(self, x):
    #     x = F.normalize(x, dim=-1, p=2)
    #     cluster_features = self.cluster_queue.get_features().clone().detach()
    #     cluster_features = F.normalize(cluster_features, dim=-1, p=2)
    #     cluster_dis = torch.matmul(cluster_features, x.T).T
    #     cluster_dis = torch.sum(cluster_dis, dim=1) / self.cluster_queue.current_size
    #     return cluster_dis
    #
    # @torch.no_grad()
    # def get_soft_label(self, features):
    #     cluster_dis = self.calc_distance(features)
    #     cluster_dis = cluster_dis / self.tmp
    #     return F.softmax(cluster_dis, dim=1)

    @torch.no_grad()
    def get_soft_label(self, x):
        x = F.normalize(x, dim=-1, p=2)
        cluster_features = self.cluster_queue.get_features().clone().detach()
        cluster_features = cluster_features.reshape(-1, self.dim)
        cluster_dis = torch.einsum('b c, c n -> b n', [x, cluster_features.T])
        cluster_dis = torch.softmax(cluster_dis, dim=1).reshape(-1, self.k, self.n)
        cluster_dis = torch.sum(cluster_dis / self.tmp, dim=-1)
        # cluster_dis = torch.softmax(cluster_dis / self.tmp, dim=1)
        # cluster_dis = torch.sum(cluster_dis, dim=-1) / self.cluster_queue.current_size
        return cluster_dis

    @torch.no_grad()
    def update(self, features, labels, masks, image_inds=None):
        # record current gpu for outputs
        batch_size = features.shape[0]
        rank_id = torch.distributed.get_rank()

        # gather from all gpus
        features_all = concat_all_gather(features)
        labels_all = concat_all_gather(labels)
        masks_all = concat_all_gather(masks)
        if image_inds is not None:
            image_inds_all = concat_all_gather(image_inds)
        else:
            image_inds_all = None

        # record patch labels calculated from cluster
        cluster_targets = labels_all.clone()
        one_hot_labels_all = torch.nn.functional.one_hot(labels_all, self.k)


        # calculate patch's probability belongs to each cluster
        soft_prob = self.get_soft_label(features_all)
        masked_prob = soft_prob * masks_all

        wsi_prob = torch.sum(masked_prob, axis=1)

        update_ind = []
        # search negative patch and false positive patch
        fp_found_ind = (wsi_prob > self.t_neg) & (labels_all > 0)
        neg_update_ind = torch.where(labels_all == 0)[0]
        update_ind.append(neg_update_ind[:self.top_k])
        # update_ind.append(neg_update_ind[-self.top_k:])
        # update_ind.append(neg_update_ind)

        self.label_num += labels_all.bincount(minlength=self.k).cpu()
        # record for cluster update information, not necessary
        if masked_prob[fp_found_ind].size(0) > 0:
            corrected_targets = torch.argmax(masked_prob[fp_found_ind], axis=1)
            self.corrected_out_num += labels_all[fp_found_ind].bincount(minlength=self.k).cpu()
            self.corrected_in_num += corrected_targets.bincount(minlength=self.k).cpu()

            cluster_targets[fp_found_ind] = corrected_targets

        # search positive patch
        pos_prob, pos_ind = torch.sort(one_hot_labels_all * soft_prob, axis=0, descending=True)
        self.top_k_prob = pos_prob[:self.top_k].mean(axis=0).cpu()
        if pos_ind.size(0) > 0:
            update_ind.append(pos_ind[:self.top_k, 1:].reshape(-1))

        # update cluster
        if len(update_ind) > 0:
            update_ind = torch.cat(update_ind)
            self.cluster_queue.dequeue_and_enqueue(features_all[update_ind], labels_all[update_ind],
                                                   image_inds_all[update_ind])
            # record for cluster update
            self.update_num += labels_all[update_ind].bincount(minlength=self.k).cpu()

        return cluster_targets[rank_id * batch_size: (rank_id + 1) * batch_size]

    def reset_update_info(self):
        self.label_num = torch.zeros(self.k)
        self.corrected_in_num = torch.zeros(self.k)
        self.corrected_out_num = torch.zeros(self.k)
        self.top_k_prob = torch.ones(self.k)
        self.update_num = torch.zeros(self.k)

    def print_update_info(self):
        update_table = PrettyTable()
        table_header = [' '] + [str(i) for i in range(self.k)]
        update_table.field_names = table_header
        label_info = ['label num'] + [str(i) for i in self.label_num.numpy().astype(int)]
        update_table.add_row(label_info)
        crrct_out_info = ['in'] + [str(i) for i in self.corrected_in_num.numpy().astype(int)]
        update_table.add_row(crrct_out_info)
        crrct_in_info = ['out'] + [str(i) for i in self.corrected_out_num.numpy().astype(int)]
        update_table.add_row(crrct_in_info)
        top_k_prob = ['top_k prob'] + [str(i) for i in self.top_k_prob.numpy()]
        update_table.add_row(top_k_prob)
        update_num = ['update num'] + [str(i) for i in self.update_num.numpy().astype(int)]
        update_table.add_row(update_num)
        print(update_table)

    def save_cluster(self, save_path):
        cluster_features = self.cluster_queue.get_features()
        cluster_checkpoint = {
            'features': cluster_features,
            'current_size': self.cluster_queue.current_size,
            'image_inds': self.cluster_queue.image_inds
        }
        torch.save(cluster_checkpoint, save_path)

    def load_cluster(self, data_path, device=None):
        cluster_checkpoint = torch.load(data_path, map_location='cpu')
        cluster_features = cluster_checkpoint['features'].cuda(device)
        current_size = cluster_checkpoint['current_size'].cuda(device)
        if 'image_inds' in cluster_checkpoint:
            image_inds = cluster_checkpoint['image_inds'].cuda(device)
        else:
            image_inds = None
        self.cluster_queue.load_queue(cluster_features, current_size, image_inds)

    def cuda(self, device=None):
        self.cluster_queue.cuda(device)

    def visualize_cluster(self, image_list, epoch, save_path, top_k=100):
        os.makedirs(save_path, exist_ok=True)
        features = []
        labels = []
        for i in range(self.k):
            sample_inds = torch.randperm(int(self.cluster_queue.current_size[i].item()))[:top_k]
            features.append(self.cluster_queue.features_queue[i, sample_inds])
            labels.append(torch.ones(top_k) * i)

            image_inds = self.cluster_queue.image_inds[i, sample_inds]
            plot_patches(image_list, image_inds, os.path.join(save_path, '[epoch{}] cluster[{}].jpg'.format(epoch, i)))

        shuffle_inds = np.random.permutation(np.arange(self.k * top_k))
        features = torch.cat(features, dim=0).cpu().numpy()[shuffle_inds]
        labels = torch.cat(labels, dim=0).cpu().numpy()[shuffle_inds]
        plot_tsne(features, labels, os.path.join(save_path, '[epoch{}] tsne.jpg'.format(epoch)))


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
