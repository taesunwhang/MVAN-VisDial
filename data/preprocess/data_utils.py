import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import h5py
import collections

from torch.utils.data import DataLoader
from config.hparams import *

from data.preprocess.visdial_preprocess_dataset import VisDialDataset

class DataUtils(object):
  def __init__(self):
    hparams = BASE_PARAMS
    hparams["model_name"] = '%s-%s' % (hparams["encoder"], hparams["decoder"])

    self.hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    self._build_dataloader()

  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================

    self.valid_dataset = VisDialDataset(
      self.hparams,
      self.hparams.visdial_json % "val",
      self.hparams.valid_dense_json,
      overfit=self.hparams.overfit,
      return_options=True,
    )

    self.valid_dataloader = DataLoader(
      self.valid_dataset,
      batch_size=1,
      num_workers=self.hparams.cpu_workers,
    )

    self.test_dataset = VisDialDataset(
      self.hparams,
      self.hparams.visdial_json % "test",
      overfit=self.hparams.overfit,
      return_options=True,
    )

    self.test_dataloader = DataLoader(
      self.test_dataset,
      batch_size=1,
      num_workers=self.hparams.cpu_workers,
    )

    self.train_dataset = VisDialDataset(
      self.hparams,
      self.hparams.visdial_json % "train",
      overfit=self.hparams.overfit,
    )

    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=1,
      num_workers=self.hparams.cpu_workers,
    )

  def build_h5_data(self, h5_path):
    datasets = {
      "val": self.valid_dataset,
      "test": self.test_dataset,
      "train" : self.train_dataset
    }
    dataloaders = {
      "val": self.valid_dataloader,
      "test": self.test_dataloader,
      "train" : self.train_dataloader
    }

    for data_key in dataloaders.keys():
      save_h5 = h5py.File(h5_path % (data_key), "w")
      tqdm_batch_iterator = tqdm(dataloaders[data_key])

      cnt = 0
      h5_key_dict = {}
      for idx, batch in enumerate(tqdm_batch_iterator):
        if idx == 0:
          for key in batch.keys():
            key_size = [len(datasets[data_key])] + [each_shape for each_shape in batch[key][0].shape]
            h5_key_dict[key] = save_h5.create_dataset(key, tuple(key_size))
          print(h5_key_dict)

        for h5_key in h5_key_dict.keys():
          h5_key_dict[h5_key][idx] = np.array(batch[h5_key][0])

        cnt+= 1
      print(data_key, ":", cnt)
      save_h5.attrs["split"] = data_key
      save_h5.close()

if __name__ == '__main__':
    data_utils = DataUtils()

    # single_text_path = "data/visdial_1.0_text/visdial_1.0_single_text_%s.hdf5"
    multi_text_path = "data/visdial_1.0_text/visdial_1.0_multi_text_%s.hdf5"

    data_utils.build_h5_data(multi_text_path)
