import torch
import torch.nn.functional as F
from torch.nn.functional import kl_div
import os
import math
from enum import Enum
import sys


class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
#        self.current_best_accuracy_dict = {}
#        for dataset in self.datasets:
#            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

#    def is_better(self, accuracies_dict):
#        is_better = False
#        is_better_count = 0
#        for i, dataset in enumerate(self.datasets):
#            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
#                is_better_count += 1
#
#        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
#            is_better = True
#
#        return is_better

#    def replace(self, accuracies_dict):
#        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

#    def get_current_best_accuracy_dict(self):
#        return self.current_best_accuracy_dict


def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    elif test_mode:
        if not os.path.exists(checkpoint_dir):
            print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            sys.exit()


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    #if not test_mode and not resume:
    if (not resume) and (not test_mode):
        os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def LSR(labels, class_num=100, factor=0.1, device=None):                 

    #value = 1 - factor
    one_hot = torch.zeros(labels.size(0), class_num, device=device)
                    
    labels = labels.view(labels.size(0),-1)
    value_added = torch.Tensor(labels.size(0),1).fill_(1 - factor).to(device)
                                     
    one_hot.scatter_add_(1, labels, value_added)
    one_hot += factor / (class_num)
    return one_hot


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss with CE.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def ff_mi_loss(F1, F2, temperature=1.):
    """
        Before align--> F1: [M, C, T, H, W]
        After align --> F2: [N, M, C, T, H, W]
        F1 -> F2
    """
    softmax = torch.nn.Softmax(dim=-1)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    B = F1.shape[0] * F1.shape[1]
    T = F1.shape[-3]
    F1 = F1.mean(-1).mean(-1) # N, M, C, T
    F2 = F2.mean(-1).mean(-1) # N, M, C, T
    F1_org = F1.reshape(B, -1, T).permute(0,2,1)  # B, T, C
    F2_org = F2.reshape(B, -1, T).permute(0,2,1)  # B, T, C
    #F1 = F1.reshape(B*T, -1) # B*T, C
    #F2 = F2.reshape(B*T, -1) # B*T, C
    F1 = log_softmax(F1_org.detach() / temperature)
    F2 = softmax(F2_org / temperature)
    mi = kl_div(F1, F2, reduction='none') # B, T, C
    mi = mi.sum(dim=-1).sum(dim=-1).mean()

    return mi


def fy_mi_loss(test_logits_sample, test_labels, device, temperature=1.0):
    """
    Compute the MI loss between logits with kl div.
    """
    
    softmax = torch.nn.Softmax(dim=-1)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    size = test_logits_sample.size()
    #sample_count = size[0]  # scalar for the loop counter
    #num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)
    #score = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    #for sample in range(sample_count):
    #breakpoint()
    F1 = log_softmax(test_logits_sample[0].detach())
    #F2 = LSR(test_labels, class_num=5, factor=0.1, device=device).float()
    F2 = F.one_hot(test_labels, num_classes=5).float()
    score = kl_div(F1, F2, reduction='batchmean')
    return score

def yy_mi_loss(sampled_logits, dense_logits, temperature=1.0):
    """
    Compute the KLD between predicted logits using sampled frames & complete frames
    Input: sampled logits:  [1, n_queries, way];  dense logits: [1, n_queries, way]
    """
    softmax = torch.nn.Softmax(dim=-1)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    size = sampled_logits.size()
    F1 = log_softmax(sampled_logits[0].detach())
    F2 = softmax(dense_logits[0]).float()
    score = kl_div(F1, F2, reduction='batchmean') # Treat the n_queries as batch number
    return score


# def score_loss(score, k, device):
#     """
#     Constrain the output score'sum to k
#     input: way * shot, frame
#     output: loss
#     """
#     video_num = score.size(0)
#     sum_target = k * torch.ones(video_num, dtype=torch.float, device=device)
#     sum_s = score.sum(dim=1) # way*shot, 1

#     return torch.abs(sum_s - sum_target).sum() / video_num


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds

def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


### Backup for visualization func in model_sampler_proto.py

# def vis(images, indices, args):
#     '''
#     Images: way*shot, 3, w, h
#     Indices: way*shot, k, len
#     '''
#     _, c, w, h = images.size()
#     images = images.view(-1, args.seq_len, c, w, h) # way * shot, seq_len, c, w, h
#     inx = random.randint(0, images.size(0)-1)
#     images = images[inx] # seq_len, c, w, h
#     mask_t = indices[inx].sum(dim=0).view(-1, 1, 1, 1) # seq_len, 1, 1, 1
#     mask_f = 1.0 - indices[inx].sum(dim=0).view(-1, 1, 1, 1) # seq_len, 1, 1, 1

#     masked_image = mask_t * images + 0.3 * mask_f * images
#     window.images(masked_image)
#     #mask = torch.ones_like(images)

# def show_att_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255.

#     if np.max(img) > 1.5:
#         breakpoint()
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")

#     cam = heatmap + img.transpose(1,2,0)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)

# def vis_amp(org, amplify, maps):
#     '''
#     org: way*shot, 3, w, h
#     map: way*shot, w', h'
#     '''

#     org = org[0,:,:,:].detach().cpu()
#     size = org.size(-1)
#     amplify = amplify[0,:,:,:].detach().cpu()

#     maps = F.interpolate(maps.unsqueeze(1), (size, size), mode='bilinear', align_corners=True).squeeze(1)
#     img = org.numpy()
#     maps = maps[0,:,:].detach().cpu().numpy()
#     att_image = show_att_on_image(img, maps, use_rgb=True).transpose(2,0,1)
#     att_image = torch.from_numpy(att_image) / 255.

#     cat = torch.stack([org,amplify,att_image],dim=0)
#     window.images(cat)

# def vis_single(selected, amplified, maps, types='support'):
#     '''
#     seleted: 8, 3, w, h
#     amplified: 8, 3, w, h
#     maps: 8, 16, 16
#     '''   
#     num = selected.size(0)
#     size = selected.size(-1)
#     atts = torch.zeros_like(selected)
#     maps = F.interpolate(maps.unsqueeze(1), (size, size), mode='bilinear', align_corners=True).squeeze(1)
#     for i in range(num):
#         att_on_image = torch.from_numpy(show_att_on_image(selected[i].numpy(), maps[i].numpy(), use_rgb=True).transpose(2,0,1) / 255.)
#         atts[i] = att_on_image
#     cat = torch.cat([selected, atts, amplified],dim=0) # 24, 3, w, h
#     #indx = random.randint(0,100)
#     #plt.imsave('vis_att/maps/map'+ str(indx)+ '.jpg', atts[0].detach().cpu().permute(1,2,0).numpy())
#     #plt.imsave('vis_att/maps/image'+ str(indx)+ '.jpg', selected[0].detach().cpu().permute(1,2,0).numpy())
#     #plt.imsave('vis_att/maps/amp'+ str(indx)+ '.jpg', amplified[0].detach().cpu().permute(1,2,0).numpy())
#     #torch.save(maps[0], 'vis_att/maps/map'+ str(indx)+ '.pt')
#     window.images(
#         cat,
#         opts=dict(title=types),
#         nrow=8
#     )


# def vis_all(selected, amplified, maps, types='support'):
#     '''
#     seleted: way*shot*8, 3, w, h
#     amplified: way*shot*8, 3, w, h
#     maps: way*shot*8, 16, 16
#     For convience, way set to 2, shot set to 1
#     '''
#     selected = selected.detach().cpu()
#     amplified = amplified.detach().cpu()
#     maps = maps.detach().cpu()
#     size = selected.size(-1)
#     num = selected.size(0)
#     for i in range(int(num/8)):
#         vis_single(selected[(8*i):(8*(i+1)),:,:,:], amplified[(8*i):(8*(i+1)),:,:,:], maps[(8*i):(8*(i+1)),:,:], types)

# def vis_map(images, maps):
#     '''
#     images: way*shot, 3, w, h
#     map: way*shot, 16, 16
#     '''
#     maps = F.interpolate(maps.unsqueeze(1), (images.size(-1), images.size(-1)), mode='bilinear', align_corners=True).squeeze(1)
#     img = images[0,:,:,:].detach().cpu().numpy()
#     maps = maps[0,:,:].detach().cpu().numpy()
#     att_image = show_att_on_image(img, maps, use_rgb=True)
#     window.image(att_image.transpose(2,0,1))

# def vis_att(atts):
#     '''
#     att: way*shot, w, h [att1, att2, att3, att4]
#     '''
#     stack = []
#     #breakpoint()
#     for att in atts:
#         att = att[0,:,:].detach()
#         max_value = att.max()
#         min_value = att.min()
#         att = (att - min_value) / (max_value - min_value)
#         att = att.cpu().numpy()
#         #.cpu().numpy()
#         att = cv2.applyColorMap(np.uint8(255 * att), colormap=cv2.COLORMAP_JET)
#         att = cv2.cvtColor(att, cv2.COLOR_BGR2RGB)
#         att = np.float32(att) / 255
#         att = torch.from_numpy(att)
#         stack.append(att)
#     #breakpoint()
#     atts_all = torch.stack(stack, dim=0).permute(0,3,2,1) # 4, w, h, 3
#     atts_all = F.interpolate(atts_all, (224, 224), mode='bilinear', align_corners=True)
#     window.images(atts_all)

