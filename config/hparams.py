import random
from collections import defaultdict

OLD_DATASET_PARAMS = defaultdict(
  dataset_version='0.9',
  visdial_json='data/v0.9/visdial_0.9_%s.json',
)

BASE_PARAMS = defaultdict(
  # Dataset reader arguments
  dataset_version='1.0',

  img_feature_type="dan_faster_rcnn_x101", # faster_rcnn_x101, dan_faster_rcnn_x101
  model_train_type="multi", # single, multi
  img_features_h5='data/visdial_1.0_img/features_%s_%s.h5', # img_feature_type | train, val, test
  imgid2idx_path='data/visdial_1.0_img/%s_imgid2idx.pkl', # dan_img - train, val, test
  text_features_h5='data/visdial_1.0_text/visdial_1.0_%s_text_%s.hdf5',

  word_counts_json='data/v1.0/visdial_1.0_word_counts_train.json',
  glove_npy='data/visdial_1.0_text/glove.npy',
  pretrained_glove='data/word_embeddings/glove/glove.6B.300d.txt',

  visdial_json='data/v1.0/visdial_1.0_%s.json',
  valid_dense_json='data/v1.0/visdial_1.0_val_dense_annotations.json',

  # Model save arguments
  root_dir='', # for saving logs, checkpoints and result files
  save_dirpath ='checkpoints/',
  load_pthpath='',

  img_norm=True,
  max_sequence_length=20,
  vocab_min_count=5,

  # Train related arguments
  gpu_ids=[0,1],
  cpu_workers=4,
  tensorboard_step=100,
  do_vaild=True,
  overfit=False,
  random_seed=random.sample(range(1000, 10000), 1), # 3143
  concat_history=True,

  # Opitimization related arguments
  num_epochs=8,
  train_batch_size=16, # 32 x num_gpus is a good rule of thumb
  eval_batch_size=16,
  virtual_batch_size=32,
  training_splits="train",
  evaluation_type="disc",
  lr_scheduler="LambdaLR",
  warmup_epochs=2,
  warmup_factor=0.01,
  initial_lr=0.001,
  lr_gamma=0.1,
  lr_milestones=[5],  # epochs when lr —> lr * lr_gamma
  lr_gamma2=0.1,
  lr_milestones2=[6],  # epochs when lr —> lr * lr_gamma2

  # Model related arguments
  encoder='mvan',
  decoder='disc',  # [disc,gen]

  img_feature_size=2048,
  word_embedding_size=300,
  lstm_hidden_size=512,
  lstm_num_layers=2,
  dropout=0.4,
  dropout_fc=0.25,
)

MVAN_MULTI_PARAMS= BASE_PARAMS.copy()
MVAN_MULTI_PARAMS.update(
  gpu_ids=[0],
  num_epochs=8,
  train_batch_size=8, # 32 x num_gpus is a good rule of thumb
  eval_batch_size=2,
  virtual_batch_size=32,
  cpu_workers=4,

  # Model related arguments
  encoder='mvan',
  decoder='disc_gen',  # [disc,gen]
  evaluation_type="disc_gen",
  aggregation_type="average",
)