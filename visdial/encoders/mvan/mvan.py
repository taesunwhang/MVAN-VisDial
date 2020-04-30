import torch
from torch import nn
from torch.nn.functional import normalize
from visdial.utils import DynamicRNN
from .modules import ContextMatching, TopicAggregation, ModalityFusionTopic, ModalityFusionContext

class MVANEncoder(nn.Module):
	def __init__(self, hparams, vocabulary):
		super().__init__()
		self.hparams = hparams
		self.word_embed = nn.Embedding(
			len(vocabulary),
			hparams.word_embedding_size,
			padding_idx=vocabulary.PAD_INDEX,
		)
		self.ques_rnn = nn.LSTM(
			hparams.word_embedding_size,
			hparams.lstm_hidden_size,
			hparams.lstm_num_layers,
			batch_first=True,
			dropout=hparams.dropout,
			bidirectional=True
		)
		self.hist_rnn = nn.LSTM(
			hparams.word_embedding_size,
			hparams.lstm_hidden_size,
			hparams.lstm_num_layers,
			batch_first=True,
			dropout=hparams.dropout,
			bidirectional=True
		)
		self.hist_rnn = DynamicRNN(self.hist_rnn)
		self.ques_rnn = DynamicRNN(self.ques_rnn)
		self.context_matching = ContextMatching(self.hparams) # 1) Context Matching
		self.topic_aggregation = TopicAggregation(self.hparams) # 2) Topic Aggregation

		# Modality Fusion
		self.modality_fusion_topic = ModalityFusionTopic(self.hparams) # Modality Fusion Topic
		self.modality_fusion_context = ModalityFusionContext(self.hparams) # Modality Fusion Context

		# 2048 + 1024 * 2 -> 512
		fusion_size = (hparams.img_feature_size + hparams.lstm_hidden_size * 2 * 2)
		self.fusion = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			nn.Linear(fusion_size, hparams.lstm_hidden_size),
			nn.ReLU()
		)
		print(self.fusion)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, batch):
		"""Visual Features"""
		img, img_mask = self.init_img(batch) # bs, np, 2048
		_, num_p, img_feat_size = img.size()

		"""Language Features"""
		ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad = self.init_q_embed(batch)
		hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad = self.init_h_embed(batch)
		bs, num_r, bilstm = ques_encoded.size()

		"""question features reshape"""
		ques_word_embed = ques_word_embed.view(bs, num_r, -1, self.hparams.word_embedding_size)
		ques_word_encoded = ques_word_encoded.view(bs, num_r, -1, bilstm)
		ques_not_pad = ques_not_pad.view(bs, num_r, -1)

		"""dialog history features reshape"""
		hist_word_embed = hist_word_embed.view(bs, num_r, -1, self.hparams.word_embedding_size)
		hist_word_encoded = hist_word_encoded.view(bs, num_r, -1, bilstm)
		hist_not_pad = hist_not_pad.view(bs, num_r, -1)

		context_matching_feat = []
		topic_aggregation_feat = []

		for c_r in range(num_r):
			"""Context Matching"""
			accu_h_sent_encoded = hist_encoded[:, 0:c_r + 1, :]      # bs, num_r, bilstm
			curr_q_sent_encoded = ques_encoded[:, c_r:(c_r + 1), :]  # bs, 1, bilstm
			context_aware_feat, context_matching_score = self.context_matching(curr_q_sent_encoded, accu_h_sent_encoded)
			context_matching_feat.append(context_aware_feat)

			"""Topic Aggregation"""
			curr_q_word_embed = ques_word_embed[:, c_r, :, :]              # bs, sl_q, word_embed_size
			curr_q_word_encoded = ques_word_encoded[:, c_r, :, :]          # bs, sl_q, bilstm
			accu_h_word_embed = hist_word_embed[:, 0:(c_r + 1), :, :]      # bs, nr, sl_h, bilstm
			accu_h_word_encoded = hist_word_encoded[:, 0:(c_r + 1), :, :]  # bs, nr, sl_h, bilstm
			accu_h_not_pad = hist_not_pad[:, 0:(c_r + 1), :]               # bs, nr, sl_h

			topic_aware_feat = self.topic_aggregation(curr_q_word_embed, curr_q_word_encoded,
																								accu_h_word_embed, accu_h_word_encoded, accu_h_not_pad,
																								context_matching_score)
			topic_aggregation_feat.append(topic_aware_feat)

		context_matching = torch.cat(context_matching_feat, dim=1)
		topic_aggregation = torch.stack(topic_aggregation_feat, dim=1)  # bs, nr, sl_q, lstm

		"""Modality Fusion"""
		mf_topic_feat = self.modality_fusion_topic(img, topic_aggregation, ques_not_pad)          # topic-view
		mf_context_feat = self.modality_fusion_context(mf_topic_feat, context_matching, img_mask) # context-view

		multi_view_fusion = self.fusion(mf_context_feat)

		return multi_view_fusion

	def init_img(self, batch):
		img = batch['img_feat']
		"""image feature normarlization"""
		if self.hparams.img_norm:
			img = normalize(img, dim=1, p=2)
		mask = (0 != img.abs().sum(-1)).unsqueeze(1)
		return img, mask

	def init_q_embed(self, batch):
		ques = batch['ques']
		bs, nr, sl_q = ques.size()
		lstm = self.hparams.lstm_hidden_size
		"""bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
		ques_not_pad = (ques != 0).bool()
		ques_not_pad = ques_not_pad.view(-1, sl_q).unsqueeze(-1)
		ques_pad = (ques == 0).bool()
		ques_pad = ques_pad.view(-1, sl_q).unsqueeze(1)

		ques = ques.view(-1, sl_q)
		ques_word_embed = self.word_embed(ques)
		ques_word_encoded, _ = self.ques_rnn(ques_word_embed, batch['ques_len'])

		loc = batch['ques_len'].view(-1).cpu().numpy() - 1

		# sentence-level encoded
		ques_encoded_forawrd = ques_word_encoded[range(bs *nr), loc,:lstm]
		ques_encoded_backward = ques_word_encoded[:, 0,lstm:]
		ques_encoded = torch.cat((ques_encoded_forawrd, ques_encoded_backward), dim=-1)
		ques_encoded = ques_encoded.view(bs, nr, -1)

		return ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad

	def init_h_embed(self, batch):
		hist = batch['hist']
		bs, nr, sl_h = hist.size()
		lstm = self.hparams.lstm_hidden_size
		"""bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
		hist_not_pad = (hist != 0).bool()
		hist_not_pad = hist_not_pad.view(-1, sl_h).unsqueeze(-1)
		hist_pad = (hist == 0).bool()
		hist_pad = hist_pad.view(-1, sl_h).unsqueeze(1)

		hist = hist.view(-1, sl_h)  # bs*nr,sl_q
		hist_word_embed = self.word_embed(hist)  # bs*nr,sl_q, emb_s
		hist_word_encoded, _ = self.hist_rnn(hist_word_embed, batch['hist_len'])

		loc = batch['hist_len'].view(-1).cpu().numpy()-1

		# sentence-level encoded
		hist_encoded_forawrd = hist_word_encoded[range(bs * nr), loc, :lstm]
		hist_encoded_backward = hist_word_encoded[:, 0, lstm:]
		hist_encoded = torch.cat((hist_encoded_forawrd, hist_encoded_backward), dim=-1)
		hist_encoded = hist_encoded.view(bs, nr, -1)

		return hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad