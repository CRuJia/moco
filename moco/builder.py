import torch
import torch.nn as nn
import torchvision.models as models
from ipdb import set_trace


class MoCo(nn.Module):
    """
    Build a Moco
    """

    def __init__(self, base_encoder, dim=128, K=1024, m=0.999, T=0.01, mlp=False):
        """
        base_encoder: backebone, here is resnet50
        dim: feature dimension (default 128)
        K: queue size, number of negative keys (default 1024, origin paper is 65536)
        m: moco momentum of updating key encoder (default 0.999)
        T: softmax temperature (default 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn((dim, K)))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute the query feature
        q = self.encoder_q(im_q)  # queries:NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = k[idx_unshuffle]
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+k)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply tmeprature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


if __name__ == "__main__":
    resnet50 = models.__dict__["resnet50"]
    moco = MoCo(resnet50, mlp=True)
    # im_q  = torch.randn(2,3,500,500)
    # im_k  = torch.randn(2,3,500,500)
    # logits, labels = moco(im_q, im_k)
    # set_trace()
