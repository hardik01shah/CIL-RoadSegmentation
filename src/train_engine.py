import torch
import torch.nn as nn
import wandb
import math
import logging
import sys
import os

from utils.metrics import (
    segmentation_metrics,
    update_pr_curve,
    compute_pr_curve,
    get_best_threshold,
    plot_pr_curve
)

class TrainEngine:
    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        config,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Logging
        self.global_step = 0
        self.epoch = None
        self.train_logs = {}
        self.val_logs = {}
        self.init_logging(phase='train')
        self.init_logging(phase='val')

        # Metrics
        self.best_threshold = 0.5
        num_thresholds = self.config['metrics']['num_thresholds']
        if num_thresholds is None:
            num_thresholds = 10
        self.thresholds = torch.linspace(0.05, 1, num_thresholds)
    
    def init_logging(self, phase='train'):
        """
        Initialize the logs for the given phase.
        """
        if phase == 'train':
            self.train_logs = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'miou': 0.0,
                'iterations': 0
            }
        elif phase == 'val':
            self.val_logs = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'miou': 0.0,
                'iterations': 0
            }
        else:
            raise ValueError(f"Invalid phase: {phase}")
    
    def train_one_epoch(self, epoch):
        self.model.train(True)
        self.epoch = epoch

        for data_iter_step, data in enumerate(self.train_loader):

            # Get the inputs and labels
            image = data['image']
            gt = data['gt']
            image = image.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(image)

            # Compute the loss
            loss = self.criterion(pred, gt, self.config)
            if not math.isfinite(loss.item()):
                logging.error(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)
            
            # Convert prediction to [0-1] heatmap using sigmoid
            pred = nn.functional.sigmoid(pred)

            # Compute metrics
            metrics = segmentation_metrics(pred, gt, self.config)

            self.train_logs['loss'] += loss.item()
            self.train_logs['iterations'] += 1
            self.global_step += 1
            for key in metrics:
                self.train_logs[key] += metrics[key]

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log the metrics
            if self.global_step % self.config['train']['log_interval'] == 0:
                self.log(
                    phase='train',
                    global_step=self.global_step
                )

            # For the last batch, log the predictions and ground truth
            if data_iter_step == len(self.train_loader) - 1:
                self.log_predictions(image, gt, pred, phase='train')

    def log_predictions(self, image, gt, pred, phase):
        """
        Log the predictions, thresholded predictions, ground truth and images to wandb.
        """

        threshold = self.config['metrics']['threshold']

        imgs = []
        heatmaps = []
        pred_t = [] # Thresholded predictions with given threshold
        pred_b = [] # Thresholded predictions with best threshold
        gts = []

        img = (image.detach().cpu().numpy()*255).astype('uint8').transpose(0, 2, 3, 1)
        gt = (gt.detach().cpu().numpy()*255).astype('uint8').transpose(0, 2, 3, 1)
        ph = (pred.detach().cpu().numpy()*255).astype('uint8').transpose(0, 2, 3, 1)
        pt = ((pred > threshold).detach().cpu().numpy()*255).astype('uint8').transpose(0, 2, 3, 1)
        pb = ((pred > self.best_threshold).detach().cpu().numpy()*255).astype('uint8').transpose(0, 2, 3, 1)

        num_imgs = min(image.size(0), self.config['logging']['num_images'])
        for i in range(num_imgs):
            imgs.append(wandb.Image(img[i]))
            heatmaps.append(wandb.Image(ph[i]))
            pred_t.append(wandb.Image(pt[i]))
            pred_b.append(wandb.Image(pb[i]))
            gts.append(wandb.Image(gt[i]))
        
        wandb.log({f"{phase}/images": imgs}, step=self.global_step)
        wandb.log({f"{phase}/heatmaps": heatmaps}, step=self.global_step)
        wandb.log({f"{phase}/pred_t": pred_t}, step=self.global_step)
        wandb.log({f"{phase}/pred_best_t": pred_b}, step=self.global_step)
        wandb.log({f"{phase}/ground_truth": gts}, step=self.global_step)
    
    def log(self, phase, global_step):
        if phase == 'train':
            logs = self.train_logs
        elif phase == 'val':
            logs = self.val_logs
        else:
            raise ValueError(f"Invalid phase: {phase}")
        
        log_text = f"Epoch: {self.epoch}, Phase: {phase}, Global step: {global_step}, "
        for key in logs:
            if key == 'iterations':
                continue
            logs[key] /= logs['iterations']
            log_text += f"{key}: {logs[key]:.4f}, "
            wandb.log({f"{phase}/{key}": logs[key]}, step=global_step)
        
        logging.info(log_text)
        self.init_logging(phase=phase)
    
    @torch.no_grad()
    def validate(self):
        """
        Validate the model.
        """

        self.model.eval()
        self.init_logging(phase='val')

        # metrics
        fp = torch.zeros(len(self.thresholds))
        tp = torch.zeros(len(self.thresholds))
        fn = torch.zeros(len(self.thresholds))
        tn = torch.zeros(len(self.thresholds))

        # get random log_idx for logging predictions
        log_idx = torch.randint(0, len(self.val_loader), (1,)).item()

        with torch.no_grad():
            for data_iter_step, data in enumerate(self.val_loader):

                # Get the inputs and labels
                image = data['image']
                gt = data['gt']
                image = image.to(self.device, non_blocking=True)
                gt = gt.to(self.device, non_blocking=True)

                # Forward pass
                pred = self.model(image)

                # Compute the loss
                loss = self.criterion(pred, gt, self.config)
                if not math.isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, stopping validation")
                    sys.exit(1)
                
                # Convert prediction to [0-1] heatmap using sigmoid
                pred = nn.functional.sigmoid(pred)

                # Compute metrics
                metrics = segmentation_metrics(pred, gt, self.config)
                cur_tp, cur_tn, cur_fp, cur_fn = update_pr_curve(pred, gt, self.thresholds)
                tp += cur_tp
                tn += cur_tn
                fp += cur_fp
                fn += cur_fn

                self.val_logs['loss'] += loss.item()
                self.val_logs['iterations'] += 1
                for key in metrics:
                    self.val_logs[key] += metrics[key]

                # Log the predictions
                if data_iter_step == log_idx:
                    self.log_predictions(image, gt, pred, phase='val')
        
        # Logging
        self.log(phase='val', global_step=self.global_step)

        # Compute the precision, recall and f1 scores for different thresholds
        precision, recall, f1 = compute_pr_curve(tp, tn, fp, fn, self.config)
        best_threshold, best_idx = get_best_threshold(f1, self.thresholds)
        self.best_threshold = best_threshold
        logging.info(f"Epoch: {self.epoch}, "
                     "Phase: val, "
                     "Step: {self.global_step}, "
                     "Best threshold: {best_threshold:.2f}, "
                     "Best Precision: {precision[best_idx]:.4f}, "
                     "Best Recall: {recall[best_idx]:.4f}, "
                     "Best F1: {f1[best_idx]:.4f}")
        wandb.log({"val/best_precision": precision[best_idx]}, step=self.global_step)
        wandb.log({"val/best_recall": recall[best_idx]}, step=self.global_step)
        wandb.log({"val/best_f1": f1[best_idx]}, step=self.global_step)

        # Log the precision-recall curve
        pr_img = plot_pr_curve(precision, recall, f1, self.thresholds)
        wandb.log({"val/pr_curve": wandb.Image(pr_img)}, step=self.global_step)

        return f1[best_idx]

    def save_model(self, save_dir, ckpt_name):
        """
        Save the model checkpoint.
        """
        to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'config': self.config
        }

        save_path = os.path.join(save_dir, ckpt_name)
        logging.info(f"Saving model checkpoint to {save_path}")

        torch.save(to_save, save_path)

    def load_model(self, model, optimizer, config, epoch):
        """
        Load the model checkpoint.
        """
        raise NotImplementedError("Loading model checkpoint is not implemented yet.")
    
        if config['train']['resume'] is None:
            return
        
        checkpoint = torch.load(config['train']['resume'])
        model.load_state_dict(checkpoint['model'])
        
        if config['mode'] == 'train':
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            logging.info(f"Loaded model checkpoint from {config['train']['resume']}")




