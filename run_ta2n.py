import torch
import numpy as np
import argparse
import os
import pickle
import sys
sys.path.append("..")
sys.path.append(".") 
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from models.model_ta2n import CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

from torch.optim import lr_scheduler
#from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random 

NUM_TEST_TASKS = 3000
PRINT_FREQUENCY = 500

TEST_ITERS = 500

def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        #self.writer = SummaryWriter()
        
        #gpu_device = 'cuda:0'
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        
        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.sch, gamma=0.9)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            print('Load checkpoint from', self.checkpoint_dir)
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN(self.args)
        # model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]

        return train_set, validation_set, test_set

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", 'ssv2_cmn', "kinetics", "hmdb", "ucf"], default="ssv2",
                            help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        #parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020,
                            help="Number of meta-training iterations.")
        parser.add_argument("--test_iters", type=int, default=500)
        parser.add_argument("--no_train", default=False, action='store_true', help='Only tesing using checkpoint')
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--checkpoint_dir", type=str, default=None, help="Save model path")
        parser.add_argument("--way", type=int, default=5, help="Way of single dataset task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context of single dataset task.")
        parser.add_argument("--query_per_class", type=int, default=5,
                            help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1,
                            help="Target samples (i.e. queries) per class used for testing.")


        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="backbone")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=10000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--img_norm", dest="img_norm", default=False, action="store_true", help="Normlize input images")
        parser.add_argument("--timewise", dest="timewise", default=False, action="store_true", help="Compute similarity in timewise")
        parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio, when set to 0.0, all values are kept")
        parser.add_argument("--dist_norm", dest="dist_norm", default=False, action="store_true", help="Add nn.NormLayer above the distance logits")


        parser.add_argument("--scratch", dest="scratch", default="/mnt/DATASET/lsy/", help="data root path")
        parser.add_argument("--gpu_id", type=str, default=1, help="GPUs ID to run on")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
       
        parser.add_argument("--split", type=int, default=3, help="Dataset split.")
        parser.add_argument('--sch', type=int, help='iters to drop learning rate', default=1000000)
        
        parser.add_argument("--metric", choices=['L2', 'cos', 'otam'], default='L2', help="Distance metric for ProtypicalNet")

        args = parser.parse_args()

        print('learning rate decay scheduler', args.sch)

        #args.scratch = "/mnt/data/sjtu/"
        #args.num_gpus = 2
        #args.num_workers = 8
        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        # if (args.backbone == "resnet50") or (args.backbone == "resnet34"):
        #     args.img_size = 224
        if args.backbone == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512
        
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join("/home/sjtu/data/splits/ssv2_OTAM/")
            args.path = os.path.join(args.scratch, "SSv2/jpg")
            args.classInd = '/home/sjtu/data/SSv2/labels/classInd.json'
        if args.dataset == 'ssv2_cmn':
            args.traintestlist = os.path.join("/home/sjtu/data/splits/ssv2_CMN/")
            args.path = os.path.join(args.scratch, "SSv2/jpg")
            args.classInd = '/home/sjtu/data/SSv2/labels/CMN_split/classInd_cmn.json'
        elif args.dataset == 'hmdb':
            args.traintestlist = os.path.join("/home/sjtu/data/splits/hmdb_ARN/")
            args.path = os.path.join(args.scratch, "HMDB51/jpg")
            args.classInd = None
        elif args.dataset == 'ucf':
            args.traintestlist = os.path.join("/home/sjtu/data/splits/ucf_ARN/")
            args.path = os.path.join(args.scratch, "UCF101/jpg")
            args.classInd = None
        elif args.dataset == 'kinetics':
            args.traintestlist = os.path.join("/home/sjtu/data/splits/kinetics_CMN/")
            args.path = os.path.join(args.scratch, "kinetics/Kinetics_frames")
            args.classInd = None

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session: 
            if self.args.no_train:
                print('Conduct Testing:')
                accuracy_dict = self.test(session)
                self.test_accuracies.print(self.logfile, accuracy_dict)
                print('Evaluation Done with', NUM_TEST_TASKS, ' iteration')
            else:
                best_accuracies = 0.0
                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations

                iteration = self.start_iteration
                for task_dict in self.video_loader:
                    if iteration >= total_iterations:
                        accuracy_dict = self.test(session)
                        self.test_accuracies.print(self.logfile, accuracy_dict)
                        break
                    iteration += 1
                    #print('iteration', iteration)
                    torch.set_grad_enabled(True)

                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.scheduler.step()
                    if (iteration + 1) % PRINT_FREQUENCY == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        #self.save_checkpoint(iteration + 1)
                        self.save_checkpoint('last')


                    if ((iteration + 1) % self.args.test_iters == 0) and (iteration + 1) != total_iterations:
                        accuracy_dict = self.test(session)
                        if accuracy_dict[self.args.dataset]["accuracy"] > best_accuracies:
                            best_accuracies = accuracy_dict[self.args.dataset]["accuracy"]
                            print('Save best checkpoint in {} iter'.format(iteration))
                            self.save_checkpoint('best')
                        self.test_accuracies.print(self.logfile, accuracy_dict)

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        model_dict = self.model(context_images, context_labels, target_images)
        target_logits = model_dict['logits']

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def test(self, session):
        self.model.eval()
        with torch.no_grad():

                self.video_loader.dataset.train = False
                accuracy_dict ={}
                accuracies = []
                iteration = 0
                item = self.args.dataset
                for task_dict in self.video_loader:
                    if iteration >= NUM_TEST_TASKS:
                        break
                    iteration += 1

                    context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
                    model_dict = self.model(context_images, context_labels, target_images)
                    target_logits = model_dict['logits']
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    current_accuracy = np.array(accuracies).mean() * 100.0
                    print('current acc:{:0.3f} in iter:{:n}'.format(current_accuracy, iteration), end='\r',flush=True)
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
                self.video_loader.dataset.train = True
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(iteration)))
        #torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint_best.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
