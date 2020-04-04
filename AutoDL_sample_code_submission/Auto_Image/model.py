# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import threading
import random

import tensorflow as tf
import torch
import torchvision as tv
import numpy as np

import skeleton
from architectures.resnet import ResNet9, ResNet18
from skeleton.projects import LogicModel, get_logger
from skeleton.projects.others import AUC, five_crop

torch.backends.cudnn.benchmark = True
threads = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in threads]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# LOGGER = get_logger(__name__)


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Model(LogicModel):
    def __init__(self, metadata):
        super(Model, self).__init__(metadata)
        self.use_test_time_augmentation = False
        self.update_transforms = False

    def build(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        in_channels = self.info['dataset']['shape'][-1]
        num_class = self.info['dataset']['num_class']

        [t.join() for t in threads]

        self.device = torch.device('cuda', 0)
        self.session = tf.Session()

        Network = ResNet18
        self.model = Network(in_channels, num_class)
        self.model_pred = Network(in_channels, num_class).eval()
        self.model_9 = ResNet9(in_channels, num_class)
        self.model_9_pred = ResNet9(in_channels, num_class).eval()

        if Network in [ResNet9, ResNet18]:
            model_path = os.path.join(base_dir, 'models')

            self.model.init(model_dir=model_path, gain=1.0)
            self.model_9.init(model_dir=model_path, gain=1.0)
        else:
            self.model.init(gain=1.0)
            self.model_9.init(gain=1.0)

        self.model = self.model.to(device=self.device, non_blocking=True)  # .half()
        self.model_pred = self.model_pred.to(device=self.device, non_blocking=True)  # .half()
        self.model_9 = self.model_9.to(device=self.device, non_blocking=True)  # .half()
        self.model_9_pred = self.model_9_pred.to(device=self.device, non_blocking=True)  # .half()
        self.is_half = self.model._half


    def update_model(self):
        num_class = self.info['dataset']['num_class']

        epsilon = min(0.1, max(0.001, 0.001 * pow(num_class / 10, 2)))
        if self.is_multiclass():
            self.model.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.tau = 8.0
        else:
            self.model.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            self.tau = 8.0
        self.model_pred.loss_fn = self.model.loss_fn
        self.init_opt()

    def init_opt(self):
        steps_per_epoch = self.hyper_params['dataset']['steps_per_epoch']
        batch_size = self.hyper_params['dataset']['batch_size']

        params = [p for p in self.model.parameters() if p.requires_grad]
        params_fc = [p for n, p in self.model.named_parameters() if
                     p.requires_grad and 'fc' == n[:2] or 'conv1d' == n[:6]]
        params_not_fc = [p for n, p in self.model.named_parameters() if
                         p.requires_grad and not ('fc' == n[:2] or 'conv1d' == n[:6])]
        init_lr = self.hyper_params['optimizer']['lr']
        warmup_multiplier = 2.0
        lr_multiplier = max(0.5, batch_size / 32)
        scheduler_lr = skeleton.optim.get_change_scale(
            skeleton.optim.gradual_warm_up(
                skeleton.optim.get_reduce_on_plateau_scheduler(
                    init_lr * lr_multiplier / warmup_multiplier,
                    patience=10, factor=.5, metric_name='train_loss'
                ),
                warm_up_epoch=5,
                multiplier=warmup_multiplier
            ),
            init_scale=1.0
        )
        self.optimizer_fc = skeleton.optim.ScheduledOptimizer(
            params_fc,
            torch.optim.SGD,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.00025,
            nesterov=True
        )
        self.optimizer = skeleton.optim.ScheduledOptimizer(
            params,
            torch.optim.SGD,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.00025,
            nesterov=True
        )

    def adapt(self, remaining_time_budget=None):
        epoch = self.info['loop']['epoch']
        input_shape = self.hyper_params['dataset']['input']
        height, width = input_shape[:2]
        batch_size = self.hyper_params['dataset']['batch_size']

        train_score = np.average([c['train']['score'] for c in self.checkpoints[-5:]])
        valid_score = np.average([c['valid']['score'] for c in self.checkpoints[-5:]])

        self.use_test_time_augmentation = self.info['loop']['test'] > 3

        if self.hyper_params['conditions']['use_fast_auto_aug']:
            self.hyper_params['conditions']['use_fast_auto_aug'] = valid_score < 0.995

        if self.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.995 or self.info['terminate']) and \
                remaining_time_budget > 120 and \
                valid_score > 0.01 and \
                self.dataloaders['valid'] is not None and \
                not self.update_transforms:
            self.update_transforms = True
            self.last_predict = None
            self.info['terminate'] = True

            original_valid_policy = self.dataloaders['valid'].dataset.transform.transforms
            policy = skeleton.data.augmentations.autoaug_policy()

            num_policy_search = 100
            num_sub_policy = 3
            num_select_policy = 5
            searched_policy = []
            for policy_search in range(num_policy_search):
                selected_idx = np.random.choice(list(range(len(policy))), num_sub_policy)
                selected_policy = [policy[i] for i in selected_idx]

                self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton.data.augmentations.Augmentation(
                        selected_policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device)
                ]

                metrics = []
                for policy_eval in range(num_sub_policy * 2):
                    valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'],
                                                                    self.datasets['num_valids'])
                    valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader, reduction='max')
                    metrics.append(valid_metrics)
                loss = np.max([m['loss'] for m in metrics])
                score = np.max([m['score'] for m in metrics])

                searched_policy.append({
                    'loss': loss,
                    'score': score,
                    'policy': selected_policy
                })

            flatten = lambda l: [item for sublist in l for item in sublist]

            searched_policy = [p for p in searched_policy if p['score'] > valid_score]

            if len(searched_policy) > 0:
                policy_sorted_index = np.argsort([p['score'] for p in searched_policy])[::-1][:num_select_policy]
                policy = flatten([searched_policy[idx]['policy'] for idx in policy_sorted_index])
                policy = skeleton.data.augmentations.remove_duplicates(policy)

                original_train_policy = self.dataloaders['train'].dataset.transform.transforms
                self.dataloaders['train'].dataset.transform.transforms += [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton.data.augmentations.Augmentation(
                        policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device)
                ]

            self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy
            self.hyper_params['optimizer']['lr'] /= 2.0
            self.init_opt()
            self.hyper_params['conditions']['max_inner_loop_ratio'] *= 3
            self.hyper_params['conditions']['threshold_valid_score_diff'] = 0.00001
            self.hyper_params['conditions']['min_lr'] = 1e-8

    def activation(self, logits):
        if self.is_multiclass():
            logits = torch.sigmoid(logits)
            prediction = (logits > 0.5).to(logits.dtype)
        else:
            logits = torch.softmax(logits, dim=-1)
            _, k = logits.max(-1)
            prediction = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1),
                                                                                                      1.0)
        return logits, prediction

    def epoch_train(self, epoch, train, model=None, optimizer=None):
        model = model if model is not None else self.model
        if epoch < 0:
            optimizer = optimizer if optimizer is not None else self.optimizer_fc
        else:
            optimizer = optimizer if optimizer is not None else self.optimizer
        model.train()
        model.zero_grad()

        num_steps = len(train)
        metrics = []
        if self.switch == True:
            self.checkpoints = []
            step = 0
            for (examples, labels, original_labels) in (self.pre_data * 2):
                logits, loss = model(examples, labels, tau=self.tau, reduction='avg')
                loss = loss.sum()
                loss.backward()

                max_epoch = self.hyper_params['dataset']['max_epoch']
                optimizer.update(maximum_epoch=max_epoch)
                optimizer.step()
                model.zero_grad()

                logits, prediction = self.activation(logits.float())
                auc = AUC(logits, original_labels.float())
                score = auc
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

                step += 1
            self.switch = False
            del self.pre_data
        else:
            for step, (examples, labels) in enumerate(train):
                if examples.shape[0] == 1:
                    examples = examples[0]
                    labels = labels[0]
                original_labels = labels
                if not self.is_multiclass():
                    labels = labels.argmax(dim=-1)

                skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits, loss = model(examples, labels, tau=self.tau, reduction='avg')
                loss = loss.sum()
                loss.backward()

                max_epoch = self.hyper_params['dataset']['max_epoch']
                optimizer.update(maximum_epoch=max_epoch)
                optimizer.step()
                model.zero_grad()
                if self.info['loop']['epoch'] < 2:
                    self.pre_data.append((examples, labels, original_labels))

                logits, prediction = self.activation(logits.float())
                auc = AUC(logits, original_labels.float())
                score = auc
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })


        train_loss = np.average([m['loss'] for m in metrics])
        train_score = np.average([m['score'] for m in metrics])
        optimizer.update(train_loss=train_loss)

        return {
            'loss': train_loss,
            'score': train_score,
        }

    def epoch_valid(self, epoch, valid, reduction='avg'):
        test_time_augmentation = False
        self.model.eval()
        num_steps = len(valid)
        metrics = []
        tau = self.tau

        with torch.no_grad():
            for step, (examples, labels) in enumerate(valid):
                original_labels = labels
                if not self.is_multiclass():
                    labels = labels.argmax(dim=-1)

                batch_size = examples.size(0)

                if self.use_test_time_augmentation and test_time_augmentation:
                    examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                logits, loss = self.model(examples, labels, tau=tau, reduction=reduction)

                if self.use_test_time_augmentation and test_time_augmentation:
                    logits1, logits2 = torch.split(logits, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits.float())
                if reduction == 'avg':
                    auc = AUC(logits, original_labels.float())
                else:
                    auc = max([AUC(logits[i:i + 16], original_labels[i:i + 16].float()) for i in
                               range(int(len(logits)) // 16)])

                score = auc
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

            if reduction == 'avg':
                valid_loss = np.average([m['loss'] for m in metrics])
                valid_score = np.average([m['score'] for m in metrics])
            elif reduction in ['min', 'max']:
                valid_loss = np.min([m['loss'] for m in metrics])
                valid_score = np.max([m['score'] for m in metrics])
            else:
                raise Exception('not support reduction method: %s' % reduction)
        self.optimizer.update(valid_loss=np.average(valid_loss))

        return {
            'loss': valid_loss,
            'score': valid_score,
        }

    def skip_valid(self, epoch):
        return {
            'loss': 99.9,
            'score': epoch * 1e-4,
        }

    def prediction(self, dataloader, model=None, test_time_augmentation=True, detach=True, num_step=None):
        tau = self.tau
        if model is None:
            model = self.model_pred

            best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
            best_loss = self.checkpoints[best_idx]['valid']['loss']
            best_score = self.checkpoints[best_idx]['valid']['score']

            states = self.checkpoints[best_idx]['model']
            model.load_state_dict(states)

        num_step = len(dataloader) if num_step is None else num_step

        model.eval()
        with torch.no_grad():
            predictions = []
            for step, (examples, labels) in zip(range(num_step), dataloader):
                batch_size = examples.size(0)
                height = int(examples.size(2) * 3 / 4)
                width = int(examples.size(3) * 3 / 4)
                if self.use_test_time_augmentation and test_time_augmentation:
                    examples1 = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
                    logits_1 = model(examples1, tau=tau)
                    logits1, logits2 = torch.split(logits_1, batch_size, dim=0)
                    logits = (logits1 + logits2) / 2.0
                else:
                    logits = model(examples, tau=tau)
                logits, prediction = self.activation(logits)

                if detach:
                    predictions.append(logits.detach().float().cpu().numpy())
                else:
                    predictions.append(logits)

            if detach:
                predictions = np.concatenate(predictions, axis=0).astype(np.float)
            else:
                predictions = torch.cat(predictions, dim=0)
        return predictions
