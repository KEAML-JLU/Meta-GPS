import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import f1_score
from learner import Learner, Scaling, Translation, Transform
from utils import sgc_precompute

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Meta(nn.Module):
    def __init__(self, args, config, config_transform, config_scal, config_trans, feat, label_num, adj, adj_tilde, adj_two, id_by_class):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.feat = feat
        self.id = id_by_class
        self.k = args.k
        self.adj = adj
        self.adj_tilde = adj_tilde # one-hop adj
        self.adj_two = adj_two # two-hop adj

        self.MI = True  
        self.hidden = args.hidden
        self.mlp = MLP(self.hidden)
        fc_params = nn.Linear(self.hidden, self.n_way, bias=None)
        self.fc = [fc_params.weight.detach()] * self.task_num
        for i in range(self.task_num): self.fc[i].requires_grad = True

        self.net = Learner(config)
        self.net = self.net.to(device)

        self.scaling = Scaling(config_scal, args, label_num)
        self.scaling = self.scaling.to(device)

        self.translation = Translation(config_trans, args, label_num)
        self.translation = self.translation.to(device)
        
        self.transformation = Transform(config_transform)
        self.transformation = self.transformation.to(device)

        self.meta_optim = optim.Adam([{'params':self.net.parameters()}, {'params':self.mlp.trans.parameters()},
                                      {'params':self.scaling.parameters()}, {'params':self.translation.parameters()},{'params':self.transformation.parameters()}], lr=self.meta_lr)
        
    def reset_fc(self):
        self.fc = [torch.Tensor(self.n_way, self.hidden)]*self.task_num

    def prework(self, meta_information):
        return self.mlp(meta_information)

    def preforward(self, support, fc):
        return F.linear(support, fc, bias=None)

    def forward(self, x_spt, y_spt, x_qry, y_qry, meta_information_dict, class_selected, labels, training):

        self.h = self.transformation(self.feat) # [25000,16]
        self.h = sgc_precompute(self.h, self.adj_tilde, self.adj_two)
        self.id_by_class_prototype_embedding = {k: self.h[np.array(self.id[k])].mean(0) for k in self.id.keys()}
        step = self.update_step if training is True else self.update_step_test
        querysz = self.n_way * self.k_qry
        losses_s = [0 for _ in range(step)]
        losses_q = [0 for _ in range(step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(step + 1)]
        f1s = [0 for _ in range(step + 1)]
        meta_information_dict = {}
        for i in range(self.task_num):
            meta_information_dict[i] = torch.stack([self.id_by_class_prototype_embedding[int(k)] for k in class_selected[i]]).to(device)

        for i in range(self.task_num):
            meta_information = meta_information_dict[i] # [n_way, hidden]
            self.fc[i] = self.prework(meta_information) # [n_way, hidden]
            logits_two = self.preforward(self.h[x_spt[i]], self.fc[i]) # the meta information of x_support
            logits_three = self.preforward(self.h[x_qry[i]], self.fc[i]) # the meta information of x_query
            
            logits_value = self.net(logits_two, vars=None)#[x_spt[i]] # logits_value is intermediate variable
            
            scaling = self.scaling(logits_value)
            translation = self.translation(logits_value)
            adapted_prior = []
            for s in range(len(scaling)):
                adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
            logits = self.net(logits_two, adapted_prior)

            loss = F.cross_entropy(logits, y_spt[i]) #+ (h_theta_update - h_theta) * 0.001
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, adapted_prior)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapted_prior)))

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(logits_three, adapted_prior)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q
                f1s[0] = f1s[0] + f1_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(logits_three, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q
                f1s[1] = f1s[1] + f1_q

            for k in range(1, step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(logits_two, fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # this is modify
                logits_q = self.net(logits_three, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))

                if training == True:
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * 0.0001

                    losses_q[k + 1] += (loss_q + l2_loss)
                else:
                    losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q
                f1s[k + 1] = f1s[k + 1] + f1_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / self.task_num
        if training == True:
            if torch.isnan(loss_q):
                pass
            else:
            # optimize theta parameters
                self.meta_optim.zero_grad()
                loss_q.backward(retain_graph=True)
                self.meta_optim.step()

        accs = np.array(corrects) / (self.task_num * querysz)
        f1_sc = np.array(f1s) / (self.task_num)

        return accs, f1_sc


class MLP(nn.Module):
    def __init__(self, hid):  # n_way = feature.shape[1]
        super(MLP, self).__init__()
        self.hidden = hid
        self.trans = nn.Linear(self.hidden, self.hidden)


    def forward(self, inputs): # inputs:[n_way, features.size(0)]
        params = self.trans(inputs)
        params = F.normalize(params, dim=-1)
        return params
