from __future__ import division
from __future__ import print_function

import time
import argparse


from utils import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=400,
                    help='Number of episodes to train.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing',
                    help='Dataset:Amazon_clothing/Amazon_eletronics/dblp/CoraFull/ogb-product/reddit')
parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--hop', type=int, help='number of neighbor', default=2)
parser.add_argument('--tem', type=float, help='the temperature of scl', default=0.5)

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
print("this dataset is ", dataset)
adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class, adj_noloop, adj_noloop_two = load_data(
    dataset) if args.dataset in ('Amazon_clothing', 'Amazon_eletronics', 'dblp') else load_intrisitic_data(args.dataset, args.seed)

labels_num = len(set(labels.cpu().numpy()))


def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


#features = sgc_precompute(features, adj, 2)

print(features.shape)

config = [('Linear', [args.way, args.way])]
config_fea = [('Linear', [features.shape[1], args.hidden])]
config_scal = [('Linear', [args.way * (args.way + 1), args.way])]
config_trans = [('Linear', [args.way * (args.way + 1), args.way])]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_num = args.task_num
features_np = features.cpu().numpy()

id_by_class_prototype_embedding = {k: features_np[np.array(id_by_class[k])].sum(0) for k in id_by_class.keys()}

maml = Meta(args, config, config_fea, config_scal, config_trans, features, labels_num, adj, adj_noloop, adj_noloop_two, id_by_class).to(device)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()


def train(class_selected, id_support, id_query):

    maml.train()
    y_spt = {}
    y_qry = {}
    meta_information = {}

    for i in range(task_num):
        y_spt[i] = torch.LongTensor([class_selected[i].index(j) for j in labels[id_support[i]]]).to(device)
        y_qry[i] = torch.LongTensor([class_selected[i].index(j) for j in labels[id_query[i]]]).to(device)
        label_list = []
        for k in labels[id_support[i]]:
            if k not in label_list:
                label_list.append(int(k))

        meta_information[i] = torch.FloatTensor(np.array([id_by_class_prototype_embedding[k] for k in label_list])).to(device)
    for epoch in range(10):
        acc, f1 = maml(id_support, y_spt, id_query, y_qry, meta_information, class_selected, labels, training=True)
    return acc, f1


def test(class_selected, id_support, id_query):

    maml.eval()
    y_spt = {}
    y_qry = {}
    meta_information = {}

    for i in range(task_num):
        y_spt[i] = torch.LongTensor([class_selected[i].index(j) for j in labels[id_support[i]]]).to(device)
        y_qry[i] = torch.LongTensor([class_selected[i].index(j) for j in labels[id_query[i]]]).to(device)
        label_list = []
        for k in labels[id_support[i]]:
            if k not in label_list:
                label_list.append(int(k))
        meta_information[i] = torch.from_numpy(np.array([id_by_class_prototype_embedding[k] for k in label_list])).to(device)
    #for epoch in range(5):
    acc, f1 = maml(id_support, y_spt, id_query, y_qry, meta_information, class_selected, labels, training=False)
    return acc, f1


if __name__ == '__main__':

    n_way = args.way
    k_shot = args.shot
    n_query = args.qry
    meta_test_num = 50
    meta_valid_num = 50

    # Sampling a pool of tasks for validation/testing
    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query, task_num) for i in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query, task_num) for i in
                 range(meta_test_num)]

    # Train model
    t_total = time.time()
    meta_train_acc = []
    best_test_acc = 0
    best_test_f1 = 0
    best_episode = 0
    best_loc = 0
    paticience = 0

    for episode in range(args.episodes):
        id_support, id_query, class_selected = \
            task_generator(id_by_class, class_list_train, n_way, k_shot, n_query, task_num)
        acc_train, _ = train(class_selected, id_support, id_query)
        meta_train_acc.append(acc_train)
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))
            
            
            # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                        np.array(meta_test_f1).mean(axis=0)))
            
            
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            paticience += 1
            if paticience > 20:
                break
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            acc = np.array(meta_test_acc).mean(axis=0)
            f1 = np.array(meta_test_f1).mean(axis=0)
            print("Meta-Test_Accuracy: {}".format(acc))
            max_test_acc, max_id = np.max(acc), np.argmax(acc)
            if max_test_acc > best_test_acc:
                best_test_acc = max_test_acc
                best_episode = episode
                best_loc = max_id
                best_test_f1 = f1[best_loc]
                paticience = 0
            print("Meta-Test_F1: {}".format(f1))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("{} way {} shot best test acc is {} and best f1 is {}, at {} episode".format(n_way, k_shot, best_test_acc, best_test_f1, best_episode))
