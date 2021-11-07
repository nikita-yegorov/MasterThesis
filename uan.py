import yaml
import easydict
import os
from os.path import join
#from google.colab import drive
import argparse


from torchvision import models
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
#from PIL import Image


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f')
parser.add_argument('--config', type=str, default='office-train-config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.safe_load(open(config_file))

save_config = yaml.safe_load(open(config_file))

args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['amazon', 'dslr', 'webcam'],
    files=[
        'amazon_reorgnized.txt',
        'dslr_reorgnized.txt',
        'webcam_reorgnized.txt'
    ],
    prefix=args.data.dataset.root_path)




source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]

a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes


source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))


classes = source_train_ds.labels
#print(sorted(classes))
freq = Counter(classes)
#print(freq)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}
#print(class_weight)

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))


source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def outlier(each_target_share_weight):
    return each_target_share_weight < args.test.w_0




# class BaseFeatureExtractor(nn.Module):
#     def forward(self, *input):
#         pass

#     def __init__(self):
#         super(BaseFeatureExtractor, self).__init__()

#     def output_num(self):
#         pass

#     def train(self, mode=True):
#         # freeze BN mean and std
#         for module in self.children():
#             if isinstance(module, nn.BatchNorm2d):
#                 print('batchnorm')
#                 module.train(False)
#             else:
#                 module.train(mode)


class ResNet50Fc(nn.Module):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=args.model.pretrained_model, normalize=True, canny=False):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                #print(args.model.pretrained_model)
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.canny = canny
        print(f'CannyRegime {self.canny}')
        print(f'NormalizeRegime {self.normalize}')
        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.canny:
            x = torch.mean(x, axis=1)
            xx = x.cpu().detach().numpy()
            edges1 = np.array([feature.canny(i) for i in xx])        
            edges1 = np.stack((edges1,)*3, axis=1)
            x = torch.from_numpy(edges1)
            x = x.to(output_device, dtype=torch.float)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y


cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()
torch.multiprocessing.freeze_support()
# if args.misc.gpus < 1:
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     gpu_ids = []
#     output_device = torch.device('cpu')
# else:
#     gpu_ids = select_GPUs(args.misc.gpus)
#     output_device = gpu_ids[0]
gpu_ids = [0]
output_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# output_device = torch.device('cpu')
print(output_device)
#exit()
now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'
print(log_dir)
logger = SummaryWriter(log_dir)

# with open(join(log_dir, 'config.yaml'), 'w') as f:
#     f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
}


class TotalNet(nn.Module):
    def __init__(self, canny):
        super(TotalNet, self).__init__()
        self.feature_extractor = ResNet50Fc(canny=canny)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0

canny=False

totalNet = TotalNet(canny=canny)

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)




# feature_extractor = totalNet.feature_extractor.to(output_device).train(True)
# classifier = totalNet.classifier.to(output_device).train(True)
# discriminator = totalNet.discriminator.to(output_device).train(True)
# discriminator_separate = totalNet.discriminator_separate.to(output_device).train(True)

# test_int = 0
# train_min_step = 5000


# data = torch.load(open('log/Nov06_10-07-11/best.pkl', 'rb'), map_location=output_device)
# feature_extractor.load_state_dict(data['feature_extractor'])
# classifier.load_state_dict(data['classifier'])
# discriminator.load_state_dict(data['discriminator'])
# discriminator_separate.load_state_dict(data['discriminator_separate'])



def train():

    # ===================optimizer
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_discriminator_separate = OptimWithSheduler(
        optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)

    global_step = 0
    best_acc = 0

    total_steps = tqdm(range(args.train.min_step),desc='global step')
    epoch_id = 0

    acc = []
    total_loss = []

    while global_step < args.train.min_step:

        iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
        epoch_id += 1


        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):


            save_label_target = label_target  # for debug usage

            label_source = label_source.to(output_device)
            label_target = label_target.to(output_device)
            label_target = torch.zeros_like(label_target)

            # =========================forward pass
            im_source = im_source.to(output_device)
            im_target = im_target.to(output_device)


            fc1_s = feature_extractor.forward(im_source)
            fc1_t = feature_extractor.forward(im_target)


            fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
            fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

            domain_prob_discriminator_source = discriminator.forward(feature_source)
            domain_prob_discriminator_target = discriminator.forward(feature_target)

            domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
            domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

            source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
            source_share_weight = normalize_weight(source_share_weight)
            target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
            target_share_weight = normalize_weight(target_share_weight)
                
            # ==============================compute loss
            adv_loss = torch.zeros(1, 1).to(output_device)
            adv_loss_separate = torch.zeros(1, 1).to(output_device)

            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)
            tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)

            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

            # ============================== cross entropy loss
            ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
            ce = torch.mean(ce, dim=0, keepdim=True)

            with OptimizerManager(
                    [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
                loss = ce + adv_loss + adv_loss_separate
                loss.backward()

            global_step += 1
            total_steps.update()

            # if global_step % args.log.log_interval == 0:
            #     counter = AccuracyCounter()
            #     counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            #     # print(counter.reportAccuracy())
            #     acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            #     # print((one_hot(label_source, len(source_classes))).shape, predict_prob_source.shape)
            #     # print(one_hot(label_source, len(source_classes)), predict_prob_source)                
            #     # print('adv_loss', adv_loss)
            #     # print('ce', ce)
            #     # print('adv_loss_separate', adv_loss_separate)
            #     # print('acc_train', acc_train)


            if global_step % args.test.test_interval == 0:

                counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
                feature_extractor.eval()
                classifier.eval()
                discriminator.eval()
                discriminator_separate.eval()

                with  torch.no_grad():
                    for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                        im = im.to(output_device)
                        label = label.to(output_device)

                        feature = feature_extractor.forward(im)
                        feature, __, before_softmax, predict_prob = classifier.forward(feature)
                        domain_prob = discriminator_separate.forward(__)

                        target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                    class_temperature=1.0)

                        for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label,
                                                                                            target_share_weight):
                            if each_label in source_classes:
                                counters[each_label].Ntotal += 1.0
                                each_pred_id = torch.argmax(each_predict_prob)
                                #print(each_label, each_pred_id)
                                if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                                    counters[each_label].Ncorrect += 1.0
                            else:
                                counters[-1].Ntotal += 1.0
                                if outlier(each_target_share_weight[0]):
                                    counters[-1].Ncorrect += 1.0
                    feature_extractor.train(True)
                    classifier.train(True)
                    discriminator.train(True)
                    discriminator_separate.train(True)


                    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
                    # for x in counters:
                    #     print(x.reportAccuracy())
                    print(acc_tests)
                    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
                    print(acc_test, np.mean(acc_tests),ce + adv_loss + adv_loss_separate)
                    acc.append(np.mean(acc_tests))
                    total_loss.append(ce + adv_loss + adv_loss_separate)

                    data = {
                    "feature_extractor": feature_extractor.state_dict(),
                    'classifier': classifier.state_dict(),
                    'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
                    'discriminator_separate': discriminator_separate.state_dict(),
                    }

                    if acc_test > best_acc:
                        best_acc = acc_test
                        with open(join(log_dir, 'best.pkl'), 'wb') as f:
                            torch.save(data, f)

                    with open(join(log_dir, "epoch.txt"), 'w') as f:
                        f.write(f"{global_step} epoch \n")
                        f.write(f'CannyRegime {canny} \n')
                        f.write(f'Best_accuracy {best_acc} \n')
                        #f.write(f'ResNet -> nn.module \n')
                        f.write("accuracy\n") 
                        np.savetxt(f, np.asarray(acc), fmt='%1.3f')

                        f.write("\ntotal_loss\n") 
                        np.savetxt(f, np.asarray(total_loss), fmt='%1.3f')

                        f.close()

                    with open(join(log_dir, 'current.pkl'), 'wb') as f:
                        torch.save(data, f)              

                #clear_output()
    return acc, total_loss



torch.cuda.empty_cache()
acc_c, loss_c = train()

