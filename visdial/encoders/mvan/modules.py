import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTrans(nn.Module):
	"""
		original code is from https://github.com/yuleiniu/rva (CVPR, 2019)
		They used tanh and sigmoid, but we used tanh and LeakyReLU for non-linear transformation function
	"""
	def __init__(self, in_dim, out_dim):
		super(GatedTrans, self).__init__()
		self.embed_y = nn.Sequential(
			nn.Linear(
				in_dim,
				out_dim
			),
			nn.Tanh()
		)
		self.embed_g = nn.Sequential(
			nn.Linear(
				in_dim,
				out_dim
			),
			nn.LeakyReLU()
		)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, x_in):
		x_y = self.embed_y(x_in)
		x_g = self.embed_g(x_in)
		x_out = x_y * x_g

		return x_out

class ContextMatching(nn.Module):
	def __init__(self, hparams):
		super(ContextMatching, self).__init__()
		self.hparams = hparams

		# non-linear transformation
		self.ques_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.lstm_hidden_size * 2,
				hparams.lstm_hidden_size
			)
		)

		self.hist_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.lstm_hidden_size * 2,
				hparams.lstm_hidden_size
			)
		)

		self.att = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			nn.Linear(hparams.lstm_hidden_size, 1),
		)

		self.softmax = nn.Softmax(dim=-1)

		self.context_gate = nn.Sequential(
			nn.Linear((hparams.lstm_hidden_size * 2) * 2,
								(hparams.lstm_hidden_size * 2) * 2),
			nn.Sigmoid()
		)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, curr_q_sent, accu_h_sent):
		"""
		Context Matching between t-th question and dialog history
		"""
		bs, nr, bilstm = accu_h_sent.size()  # hist

		curr_q_feat = self.ques_emb(curr_q_sent).repeat(1, nr, 1)
		accu_h_feat = self.hist_emb(accu_h_sent)

		att_score = self.att(curr_q_feat * accu_h_feat).squeeze(-1)  # element_wise multiplication -> attention
		att_score = self.softmax(att_score)
		hist_qatt_feat = (accu_h_sent * att_score.unsqueeze(-1)).sum(1, keepdim=True)  # weighted sum : question-relevant dialog history

		hist_ques_sent_feat = torch.cat((curr_q_sent, hist_qatt_feat), dim=-1)
		context_gate = self.context_gate(hist_ques_sent_feat)
		context_aware_feat = context_gate * hist_ques_sent_feat

		return context_aware_feat, att_score

class TopicAggregation(nn.Module):
	def __init__(self, hparams):
		super(TopicAggregation, self).__init__()
		self.hparams = hparams

		self.ques_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.lstm_hidden_size * 2,
				hparams.lstm_hidden_size
			)
		)
		self.hist_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.lstm_hidden_size * 2,
				hparams.lstm_hidden_size
			)
		)
		self.softmax = nn.Softmax(dim=-1)

		self.topic_gate = nn.Sequential(
			nn.Linear(hparams.word_embedding_size * 2, hparams.word_embedding_size * 2),
			nn.Sigmoid()
		)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, curr_q_word_embed, curr_q_word_encoded, accu_h_word_embed, accu_h_word_encoded,
							accu_h_not_pad, context_matching_score):
		"""
		Attention between ith Question and all history
		"""
		bs, sl_q, bilstm = curr_q_word_encoded.size()
		_, num_r, sl_h, _ = accu_h_word_embed.size()
		lstm = self.hparams.lstm_hidden_size
		word_embedding = self.hparams.word_embedding_size

		# non-linear transformation
		curr_q_feat = self.ques_emb(curr_q_word_encoded)
		curr_q_feat = curr_q_feat.unsqueeze(1).repeat(1,num_r,1,1).reshape(bs*num_r,sl_q,lstm)

		accu_h_feat = self.hist_emb(accu_h_word_encoded)
		accu_h_feat = accu_h_feat.reshape(bs*num_r, sl_h, lstm)

		qh_dot_score = torch.bmm(curr_q_feat, accu_h_feat.permute(0, 2, 1))

		accu_h_not_pad = accu_h_not_pad.reshape(bs*num_r,sl_h).unsqueeze(1)
		qh_score = qh_dot_score * accu_h_not_pad
		h_mask = (accu_h_not_pad.float() - 1.0) * 10000.0
		qh_score = self.softmax(qh_score + h_mask)   # bs*num_r sl_q sl_h

		# (bs*num_r sl_q sl_h 1) * (bs*num_r 1 sl_h bilstm)  => sum(dim=2) => bs*num_r sl_q bilstm
		qh_topic_att = qh_score.unsqueeze(-1) * accu_h_word_embed.reshape(bs*num_r,sl_h,word_embedding).unsqueeze(1)
		qh_topic_att = torch.sum(qh_topic_att, dim=2)
		qh_topic_att = qh_topic_att.reshape(bs,num_r,sl_q, word_embedding)

		# attention features
		hist_qatt_embed = torch.sum(context_matching_score.view(bs, num_r, 1, 1) * qh_topic_att, dim=1)

		hist_ques_word_feat = torch.cat((curr_q_word_embed, hist_qatt_embed), dim=-1)
		topic_gate = self.topic_gate(hist_ques_word_feat)  # bs, sl_q, 600
		topic_aware_feat = topic_gate * hist_ques_word_feat  # bs, sl_q, 600

		return topic_aware_feat

class ModalityFusionTopic(nn.Module):
	def __init__(self, hparams):
		super(ModalityFusionTopic, self).__init__()
		self.hparams = hparams

		self.img_nonlinear_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.img_feature_size,
				hparams.lstm_hidden_size
			),
		)

		self.topic_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.word_embedding_size * 2,
				hparams.lstm_hidden_size
			)
		)

		self.MLP = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			nn.Linear(hparams.img_feature_size + hparams.word_embedding_size * 2,
								hparams.img_feature_size + hparams.word_embedding_size * 2),
			nn.ReLU(),
			nn.Linear(hparams.img_feature_size + hparams.word_embedding_size * 2, hparams.img_feature_size), # 2648 -> 2048
			nn.ReLU(),
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, img, topic_aggregation, ques_not_pad):

		bs, num_p, img_feat_size = img.size()
		_, num_r, sl_q, topic_agg_size = topic_aggregation.size()
		lstm_size = self.hparams.lstm_hidden_size

		v_feat = self.img_nonlinear_emb(img.unsqueeze(1).repeat(1,num_r,1,1)) 	# bs np lstm
		v_feat = v_feat.reshape(bs*num_r,num_p, lstm_size)

		topic_feat = self.topic_emb(topic_aggregation)      						# bs num_r sl_q lstm
		topic_feat = topic_feat.reshape(bs*num_r, sl_q, lstm_size)

		ques_not_pad = ques_not_pad.reshape(bs*num_r, 1, sl_q)

		vt_score = torch.bmm(v_feat, topic_feat.permute(0,2,1)) # bs*num_r np sl_q
		vt_score = vt_score * ques_not_pad  # 1 or 0
		sf_mask = (ques_not_pad.float() - 1.0) * 10000.0
		vt_score = self.softmax(vt_score + sf_mask) # bs*num_r, np, sl_q

		# (bs*num_r np sl_q 1) * (bs*num_r 1 sl_q 600) => sum(dim=2) =>  bs*num_r, np, 600
		vt_att = torch.sum(vt_score.unsqueeze(-1) * topic_aggregation.reshape(bs*num_r,sl_q,topic_agg_size).unsqueeze(1), dim=2)

		# bs*num_r np img_feat
		img_expanded = img.unsqueeze(1).repeat(1,num_r,1,1).reshape(bs*num_r, num_p, img_feat_size)
		mf_topic = self.MLP(torch.cat((img_expanded, vt_att), dim=-1)) # 2048 + 600
		mf_topic = mf_topic.reshape(bs, num_r, num_p, img_feat_size)

		return mf_topic

class ModalityFusionContext(nn.Module):
	def __init__(self, hparams):
		super(ModalityFusionContext, self).__init__()

		# image
		self.mf_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.img_feature_size + hparams.lstm_hidden_size * 2 * 2,
				hparams.lstm_hidden_size
			),
		)
		self.context_matching_emb = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			GatedTrans(
				hparams.lstm_hidden_size * 2 * 2,
				hparams.lstm_hidden_size
			)
		)
		self.att = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			nn.Linear(hparams.lstm_hidden_size,1)
		)

		self.softmax = nn.Softmax(dim=-1)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, mf_topic, context_matching, img_mask=None):
		bs, num_r, num_p, mf_topic_feat_size = mf_topic.size()

		fused_feat = torch.cat((mf_topic, context_matching.unsqueeze(2).repeat(1,1,num_p,1)), dim=-1)
		mf_feat = self.mf_emb(fused_feat)

		cm_feat = self.context_matching_emb(context_matching)
		cm_feat = cm_feat.unsqueeze(2).repeat(1, 1, num_p, 1)  # bs,num_r, num_p, lstm

		att_feat = F.normalize(mf_feat * cm_feat, p=2, dim= -1)
		att_feat = self.att(att_feat).squeeze(-1)  # bs, num_r, np

		if img_mask is not None:
			att_feat = att_feat * img_mask  # 1 or 0
			sf_mask = (img_mask.float() - 1.0) * 10000.0
			att_feat += sf_mask
		att_feat = self.softmax(att_feat) # bs, num_r, np

		mf_context = torch.sum(att_feat.unsqueeze(-1) * fused_feat, dim=2) 	# bs, num_r, 4096

		return mf_context