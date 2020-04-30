import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import itertools

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from setproctitle import setproctitle
from bisect import bisect

from datetime import datetime
import numpy as np

from data.dataset import VisDialDataset
from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint

from single_evaluation import Evaluation

class MVAN(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)

    np.random.seed(hparams.random_seed[0])
    torch.manual_seed(hparams.random_seed[0])
    torch.cuda.manual_seed_all(hparams.random_seed[0])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.device = (
      torch.device("cuda", self.hparams.gpu_ids[0])
      if self.hparams.gpu_ids[0] >= 0
      else torch.device("cpu")
    )
    setproctitle(hparams.dataset_version + '_' + hparams.model_name + '_' + str(hparams.random_seed[0]))

  # def _build_data_process(self):
  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    old_split = "train" if self.hparams.dataset_version == "0.9" else None
    self.train_dataset = VisDialDataset(
      self.hparams,
      overfit=self.hparams.overfit,
      split="train",
      old_split = old_split
    )

    collate_fn = None
    if "dan" in self.hparams.img_feature_type:
      collate_fn = self.train_dataset.collate_fn

    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.train_batch_size,
      num_workers=self.hparams.cpu_workers,
      shuffle=True,
      drop_last=True,
      collate_fn=collate_fn,
    )

    print("""
      # -------------------------------------------------------------------------
      #   DATALOADER FINISHED
      # -------------------------------------------------------------------------
      """)

  def _build_model(self):

    # =============================================================================
    #   MODEL : Encoder, Decoder
    # =============================================================================

    print('\t* Building model...')
    # Pass vocabulary to construct Embedding layer.
    encoder = Encoder(self.hparams, self.train_dataset.vocabulary)
    decoder = Decoder(self.hparams, self.train_dataset.vocabulary)

    print("Encoder: {}".format(self.hparams.encoder))
    print("Decoder: {}".format(self.hparams.decoder))

    # New: Initializing word_embed using GloVe
    if self.hparams.glove_npy != '':
      encoder.word_embed.weight.data = torch.from_numpy(np.load(self.hparams.glove_npy))
      print("Loaded glove vectors from {}".format(self.hparams.glove_npy))
    # Share word embedding between encoder and decoder.
    decoder.word_embed = encoder.word_embed

    # Wrap encoder and decoder in a model.
    self.model = EncoderDecoderModel(encoder, decoder)
    self.model = self.model.to(self.device)
    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

    # =============================================================================
    #   CRITERION
    # =============================================================================
    if "disc" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss()

    elif "gen" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_dataset.vocabulary.PAD_INDEX)

    # Total Iterations -> for learning rate scheduler
    if self.hparams.training_splits == "trainval":
      self.iterations = (len(self.train_dataset) + len(self.valid_dataset)) // self.hparams.virtual_batch_size
    else:
      self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

    # =============================================================================
    #   OPTIMIZER, SCHEDULER
    # =============================================================================

    def lr_lambda_fun(current_iteration: int) -> float:
      """Returns a learning rate multiplier.

      Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
      and then gets multiplied by `lr_gamma` every time a milestone is crossed.
      """
      current_epoch = float(current_iteration) / self.iterations
      if current_epoch <= self.hparams.warmup_epochs:
        alpha = current_epoch / float(self.hparams.warmup_epochs)
        return self.hparams.warmup_factor * (1.0 - alpha) + alpha
      else:
        return_val = 1.0
        if current_epoch >= self.hparams.lr_milestones[0] and current_epoch < self.hparams.lr_milestones2[0]:
          idx = bisect(self.hparams.lr_milestones, current_epoch)
          return_val = pow(self.hparams.lr_gamma, idx)
        elif current_epoch >= self.hparams.lr_milestones2[0]:
          idx = bisect(self.hparams.lr_milestones2, current_epoch)
          return_val = self.hparams.lr_gamma * pow(self.hparams.lr_gamma2, idx)
        return return_val

    if self.hparams.lr_scheduler == "LambdaLR":
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.initial_lr)
      self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda_fun)
    else:
      raise NotImplementedError

    print(
      """
      # -------------------------------------------------------------------------
      #   Model Build Finished
      # -------------------------------------------------------------------------
      """
    )

  def _setup_training(self):
    if self.hparams.save_dirpath == 'checkpoints/':
      self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
    self.summary_writer = SummaryWriter(self.save_dirpath)
    self.checkpoint_manager = CheckpointManager(
      self.model, self.optimizer, self.save_dirpath, hparams=self.hparams
    )

    # If loading from checkpoint, adjust start epoch and load parameters.
    if self.hparams.load_pthpath == "":
      self.start_epoch = 1
    else:
      # "path/to/checkpoint_xx.pth" -> xx
      self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
      self.start_epoch += 1
      model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict)
      else:
        self.model.load_state_dict(model_state_dict)
      self.optimizer.load_state_dict(optimizer_state_dict)
      self.previous_model_path = self.hparams.load_pthpath
      print("Loaded model from {}".format(self.hparams.load_pthpath))

    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def _loss_fn(self, epoch, batch, output):
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))

    return batch_loss

  def train(self):

    self._build_dataloader()
    self._build_model()
    self._setup_training()

    # Evaluation Setup
    evaluation = Evaluation(self.hparams, model=self.model, split="val")

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    global_iteration_step = (self.start_epoch - 1) * self.iterations

    running_loss = 0.0  # New
    train_begin = datetime.utcnow()  # New
    print(
      """
      # -------------------------------------------------------------------------
      #   Model Train Starts (NEW)
      # -------------------------------------------------------------------------
      """
    )
    for epoch in range(self.start_epoch, self.hparams.num_epochs):
      self.model.train()
      # -------------------------------------------------------------------------
      #   ON EPOCH START  (combine dataloaders if training on train + val)
      # -------------------------------------------------------------------------
      combined_dataloader = itertools.chain(self.train_dataloader)

      print(f"\nTraining for epoch {epoch}:", "Total Iter:", self.iterations)
      tqdm_batch_iterator = tqdm(combined_dataloader)
      accumulate_batch = 0 # taesun New

      for i, batch in enumerate(tqdm_batch_iterator):
        buffer_batch = batch.copy()
        for key in batch:
          buffer_batch[key] = buffer_batch[key].to(self.device)

        output = self.model(buffer_batch)
        batch_loss = self._loss_fn(epoch, batch, output)
        batch_loss.backward()

        accumulate_batch += batch["img_ids"].shape[0]
        if self.hparams.virtual_batch_size == accumulate_batch \
            or i == (len(self.train_dataset) // self.hparams.train_batch_size): # last batch

          self.optimizer.step()

          # --------------------------------------------------------------------
          #    Update running loss and decay learning rates
          # --------------------------------------------------------------------
          if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
          else:
            running_loss = batch_loss.item()

          self.optimizer.zero_grad()
          accumulate_batch = 0

          self.scheduler.step(global_iteration_step)

          global_iteration_step += 1
          # torch.cuda.empty_cache()
          description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
            datetime.utcnow() - train_begin,
            epoch,
            global_iteration_step, running_loss,
            self.optimizer.param_groups[0]['lr'])
          tqdm_batch_iterator.set_description(description)

          # tensorboard
          if global_iteration_step % self.hparams.tensorboard_step == 0:
            description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
              datetime.utcnow() - train_begin,
              epoch,
              global_iteration_step, running_loss,
              self.optimizer.param_groups[0]['lr'],
              )
            self._logger.info(description)
            # tensorboard writing scalar
            self.summary_writer.add_scalar(
              "train/loss", batch_loss, global_iteration_step
            )
            self.summary_writer.add_scalar(
              "train/lr", self.optimizer.param_groups[0]["lr"], global_iteration_step
            )

      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      self.checkpoint_manager.step(epoch)
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
      self._logger.info(self.previous_model_path)

      if epoch < self.hparams.num_epochs - 1 and self.hparams.dataset_version == '0.9':
        continue

      torch.cuda.empty_cache()
      evaluation.run_evaluate(self.previous_model_path, global_iteration_step, self.summary_writer,
                              os.path.join(self.checkpoint_manager.ckpt_dirpath, "ranks_%d_valid.json" % epoch))
      torch.cuda.empty_cache()

    return self.previous_model_path