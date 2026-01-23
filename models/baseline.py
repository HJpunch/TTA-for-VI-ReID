import torch.nn as nn

from models.resnet import resnet50
from utils.calc_acc import calc_acc

from loss import CenterLoss, CenterTripletLoss, TripletLoss


class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)

        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)

        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

        if self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)

        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
            
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)

    def forward(self, inputs, labels=None, **kwargs):
        feats = self.backbone(inputs)
        feats = self.gap(feats)
        feats = feats.view(feats.shape[0], -1)

        if not self.training:
            feats = self.bn_neck(feats)
            return feats

        else:
            # bn_feat = self.bn_neck(feats)
            # return self.classifier(bn_feat), feats, bn_feat  # PaTTA 적응용
            return self.train_forward(feats, labels)

    def train_forward(self, feat, labels):
        metric = {}
        loss = 0

        if self.triplet:

            triplet_loss, *_ = self.triplet_loss(feat.float(), labels)
            loss += triplet_loss
            metric.update({'tri': triplet_loss.data})

        if self.center_loss:
            center_loss = self.center_loss(feat.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})

        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feat.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})

        feat = self.bn_neck(feat)

        if self.classification:
            logits = self.classifier(feat)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        return loss, metric