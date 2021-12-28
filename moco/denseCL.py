import torch
import torch.nn as nn
from torch.nn.modules import loss
import torchvision.models as models

from moco.necks import DenseCLNeck
from moco.heads.contrastive_head import ContrastiveHead
from ipdb import set_trace


class DenseCL(nn.Module):
    """
    DenseCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    """

    def __init__(self, base_encoder, dim=128, K=1024, m=0.999, T=0.1, loss_lambda=0.5):
        """
        base_encoder: backebone, here is resnet50
        dim: feature dimension (default 128)
        K: queue size, number of negative keys (default 1024, origin paper is 65536)
        m: moco momentum of updating key encoder (default 0.999)
        T: softmax temperature (default 0.1)
        loss_lambda: default 0.5
        """
        super(DenseCL, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.loss_lambda = loss_lambda

        # create the encoders
        # num_classes is the output fc dimension
        resnet = base_encoder(num_classes=dim, pretrained=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.encoder_q = nn.Sequential(
            self.backbone, DenseCLNeck(2048, 2048, 128, num_grid=None)
        )
        self.encoder_k = nn.Sequential(
            self.backbone, DenseCLNeck(2048, 2048, 128, num_grid=None)
        )

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = ContrastiveHead(T)
        self.init_weights()

        # create the queue
        self.register_buffer("queue", torch.randn((dim, K)))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self):
        # 这里只对denseCL neck初始化，resnet采用的是torchvision的
        self.encoder_q[1].init_weights(init_linear="kaiming")
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue2_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** only support for single GPU***
        TODO: support DDP
        """
        batch_size = x.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    def forward_train(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute the query feature
        q_b = self.encoder_q[0](im_q)  # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC, NxCxS^2, NxC
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)  # global
        q2 = nn.functional.normalize(q2, dim=1)  # dense
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle(im_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC, NxCxS^2, NxC
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)  # global
            k2 = nn.functional.normalize(k2, dim=1)  # dense
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = k[idx_unshuffle]
            k2 = k2[idx_unshuffle]
            k_grid = k_grid[idx_unshuffle]
            k_b = k_b[idx_unshuffle]

        # compute logits
        # einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2

        indexed_k_grid = torch.gather(
            k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)
        )  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        # dense positive logits:NxS^2x1
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))  # NS^2xC
        # dense negative logits
        l_neg_dense = torch.einsum("nc,ck->nk", [q_grid, self.queue2.clone().detach()])

        loss_single = self.head(l_pos, l_neg)["loss_contra"]
        loss_dense = self.head(l_pos_dense, l_neg_dense)["loss_contra"]

        losses = dict()
        losses["loss_contra_single"] = loss_single * (1 - self.loss_lambda)
        losses["loss_contra_dense"] = loss_dense * self.loss_lambda
        losses["loss"] = losses["loss_contra_single"] + losses["loss_contra_dense"]
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses

    def forward(self, im_q, im_k, mode="train"):
        if mode == "train":
            return self.forward_train(im_q, im_k)
        else:
            raise Exception("No sum mode: {}".format(mode))


if __name__ == "__main__":
    resnet50 = models.__dict__["resnet50"]
    model = DenseCL(resnet50, mlp=True).cuda()
    im_q = torch.randn(2, 3, 224, 224).cuda()
    im_k = torch.randn(2, 3, 224, 224).cuda()
    losses = model(im_q, im_k)
