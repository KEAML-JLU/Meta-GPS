import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()

        self.vars = nn.ParameterList()
        self.config = config
        for i, (name, param) in enumerate(self.config):

            if name is 'Linear':

                w = nn.Parameter(torch.ones(param[1], param[0]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

    def forward(self, features, vars=None):

        if vars is None:
            vars = self.vars

        idx = 0
        h = features.float()
        h = h.to(device)

        for name, param in self.config:
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2
        return h

    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class Scaling(nn.Module):
    def __init__(self, config_scal, args, label_num):
        super(Scaling, self).__init__()
        self.config = config_scal
        self.args = args
        self.label_dim = label_num
        self.num_attri = self.args.way
        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'Linear':
                w = nn.Parameter(torch.ones(*param))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        for name, param in self.config:
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'relu':
                x = F.relu(x)
            elif name is 'elu':
                x = F.elu(x)
            elif name is 'leaky_relu':
                x = F.leaky_relu(x)
        x = torch.mean(x, dim=0)
        x1, x2 = x[:self.args.way * self.args.way].view(self.args.way, self.args.way), x[self.args.way * self.args.way:].view(self.args.way)
        para_list = [x1, x2]
        return para_list

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class Translation(nn.Module):
    def __init__(self, config_trans, args, label_num):
        super(Translation, self).__init__()
        self.config = config_trans
        self.args = args
        self.num_attri = self.args.way
        self.label_dim = label_num
        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'Linear':
                w = nn.Parameter(torch.ones(*param))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        for name, param in self.config:
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'relu':
                x = F.relu(x)
            elif name is 'elu':
                x = F.elu(x)
            elif name is 'leaky_relu':
                x = F.leaky_relu(x)
        x = torch.mean(x, dim=0) # this is modified, in fact, it is correct
        x1, x2 = x[:self.args.way * self.args.way].view(self.args.way, self.args.way), x[self.args.way * self.args.way:].view(self.args.way)
        para_list = [x1, x2]
        return para_list

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
    

class Transform(nn.Module):
    def __init__(self, config):
        super(Transform, self).__init__()

        self.vars = nn.ParameterList()
        self.config = config
        for i, (name, param) in enumerate(self.config):

            if name is 'Linear':

                w = nn.Parameter(torch.ones(param[1], param[0]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

    def forward(self, features, vars=None):

        if vars is None:
            vars = self.vars

        idx = 0
        h = features.float()
        h = h.to(device)

        for name, param in self.config:
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2
        return h

    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
