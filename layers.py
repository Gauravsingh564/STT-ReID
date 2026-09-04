import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

class TemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        weights = F.softmax(self.net(x), dim=1)
        out = torch.sum(x * weights, dim=1)
        return out

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        with torch.amp.autocast('cuda', enabled=False):
            input = input.float()
            weight = self.weight.float()

            cosine = F.linear(F.normalize(input, eps=1e-6), F.normalize(weight, eps=1e-6))
            cosine_clamped = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))
            
            phi = cosine_clamped * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            
            return output, cosine * self.s