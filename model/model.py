import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import pad

from model.embeddings_and_logits import IndependentEmbeddingsAndLogits, IndependentEmbeddingsSharedLogits, \
    SharedEmbeddingsIndependentLogits, SharedEmbeddingsAndLogits
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder
from utils.beam_search import BeamSearch


class TransformerMT(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 joint_vocab_size: int,
                 share_embedding: bool,
                 share_projection_and_embedding: bool,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 tgt_sos_idx: int,
                 tgt_eos_idx: int,
                 positional_encoding: str,
                 emb_size: int,
                 feed_forward_size: int,
                 num_of_layers: int,
                 num_of_heads: int,
                 train_max_seq_length: int,
                 infer_max_seq_length: int,
                 infer_max_seq_length_mode: str,
                 batch_size: int,
                 update_decay: int,
                 embedding_dropout_prob: float,
                 attention_dropout_prob: float,
                 feedforward_dropout_prob: float,
                 residual_dropout_prob: float,
                 activate_function_name: str,
                 emb_norm_clip: float,
                 emb_norm_clip_type: float,
                 layer_norm_pre: str,
                 layer_norm_post: str,
                 layer_norm_encoder_start: str,
                 layer_norm_encoder_end: str,
                 layer_norm_decoder_start: str,
                 layer_norm_decoder_end: str,
                 prefix: str,
                 pretrained_src_emb: str,
                 pretrained_tgt_emb: str,
                 pretrained_src_eos: str,
                 pretrained_tgt_eos: str,
                 src_vocab: dict,
                 tgt_vocab: dict,
                 beams: int,
                 length_penalty: float,
                 criterion: nn.Module):
        super(TransformerMT, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.joint_vocab_size = joint_vocab_size

        self.share_embedding = share_embedding
        self.share_projection_and_embedding = share_projection_and_embedding

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx

        self.positional_encoding = positional_encoding

        self.emb_size = emb_size
        self.feed_forward_size = feed_forward_size
        self.num_of_layers = num_of_layers
        self.num_of_heads = num_of_heads
        self.train_max_seq_length = train_max_seq_length
        self.infer_max_seq_length = infer_max_seq_length
        self.infer_max_seq_length_mode = infer_max_seq_length_mode
        self.batch_size = batch_size

        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.feedforward_dropout_prob = feedforward_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.activate_function_name = activate_function_name

        self.emb_norm_clip = emb_norm_clip
        self.emb_norm_clip_type = emb_norm_clip_type

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post
        self.layer_norm_encoder_start = layer_norm_encoder_start
        self.layer_norm_encoder_end = layer_norm_encoder_end
        self.layer_norm_decoder_start = layer_norm_decoder_start
        self.layer_norm_decoder_end = layer_norm_decoder_end

        self.beams = beams
        self.length_penalty = length_penalty

        self.criterion = criterion

        if self.share_embedding and self.share_projection_and_embedding:
            self.embeddings = SharedEmbeddingsAndLogits(vocab_size=self.joint_vocab_size,
                                                        embedding_size=self.emb_size,
                                                        pad_idx=self.tgt_pad_idx,
                                                        max_norm=self.emb_norm_clip,
                                                        max_norm_type=self.emb_norm_clip_type)
        elif self.share_embedding:
            self.embeddings = SharedEmbeddingsIndependentLogits(vocab_size=self.joint_vocab_size,
                                                                embedding_size=self.emb_size,
                                                                pad_idx=self.tgt_pad_idx,
                                                                max_norm=self.emb_norm_clip,
                                                                max_norm_type=self.emb_norm_clip_type)
        elif self.share_projection_and_embedding:
            self.embeddings = IndependentEmbeddingsSharedLogits(src_vocab_size=self.src_vocab_size,
                                                                tgt_vocab_size=self.tgt_vocab_size,
                                                                embedding_size=self.emb_size,
                                                                src_pad_idx=self.src_pad_idx,
                                                                tgt_pad_idx=self.tgt_pad_idx,
                                                                max_norm=self.emb_norm_clip,
                                                                max_norm_type=self.emb_norm_clip_type)
        else:
            self.embeddings = IndependentEmbeddingsAndLogits(src_vocab_size=self.src_vocab_size,
                                                             tgt_vocab_size=self.tgt_vocab_size,
                                                             embedding_size=self.emb_size,
                                                             src_pad_idx=self.src_pad_idx,
                                                             tgt_pad_idx=self.tgt_pad_idx,
                                                             max_norm=self.emb_norm_clip,
                                                             max_norm_type=self.emb_norm_clip_type)

        self.encoder = TransformerEncoder(emb_size=self.emb_size,
                                          feedforward_size=self.feed_forward_size,
                                          num_of_layers=self.num_of_layers,
                                          num_of_heads=self.num_of_heads,
                                          max_seq_length=max(self.train_max_seq_length, self.infer_max_seq_length),
                                          embedding_dropout_prob=self.embedding_dropout_prob,
                                          attention_dropout_prob=self.attention_dropout_prob,
                                          positional_encoding=self.positional_encoding,
                                          residual_dropout_prob=self.residual_dropout_prob,
                                          feedforward_dropout_prob=self.feedforward_dropout_prob,
                                          activate_function_name=self.activate_function_name,
                                          layer_norm_pre=self.layer_norm_pre,
                                          layer_norm_post=self.layer_norm_post,
                                          layer_norm_start=self.layer_norm_encoder_start,
                                          layer_norm_end=self.layer_norm_encoder_end)
        self.decoder = TransformerDecoder(emb_size=self.emb_size,
                                          feedforward_size=self.feed_forward_size,
                                          num_of_layers=self.num_of_layers,
                                          num_of_heads=self.num_of_heads,
                                          max_seq_length=max(self.train_max_seq_length, self.infer_max_seq_length),
                                          embedding_dropout_prob=self.embedding_dropout_prob,
                                          attention_dropout_prob=self.attention_dropout_prob,
                                          residual_dropout_prob=self.residual_dropout_prob,
                                          feedforward_dropout_prob=self.feedforward_dropout_prob,
                                          activate_function_name=self.activate_function_name,
                                          positional_encoding=self.positional_encoding,
                                          layer_norm_pre=self.layer_norm_pre,
                                          layer_norm_post=self.layer_norm_post,
                                          layer_norm_start=self.layer_norm_decoder_start,
                                          layer_norm_end=self.layer_norm_decoder_end)

        if pretrained_src_emb != '':
            print('Load pretrained source embeddings from', pretrained_src_emb)
            with open(prefix + pretrained_src_emb, 'r') as f:
                for _, line in enumerate(f):
                    splits = line.split()
                    if len(splits) <= self.emb_size:
                        continue
                    if splits[0] == pretrained_src_eos:
                        splits[0] = '<EOS>'

                    if splits[0] in src_vocab.keys():
                        self.encoder.src_embs.weight.data.index_copy_(
                            dim=0,
                            index=torch.Tensor([src_vocab[splits[0]]]).long(),
                            source=torch.Tensor(list(float(x) for x in splits[1:])).unsqueeze(dim=0))

        if pretrained_tgt_emb != '':
            print('Load pretrained target embeddings from', pretrained_tgt_emb)
            with open(prefix + pretrained_tgt_emb, 'r') as f:
                for _, line in enumerate(f):
                    splits = line.split()
                    if len(splits) <= self.emb_size:
                        continue
                    if splits[0] == pretrained_tgt_eos:
                        splits[0] = '<EOS>'

                    if splits[0] in tgt_vocab.keys():
                        self.decoder.tgt_embs.weight.data.index_copy_(
                            dim=0,
                            index=torch.Tensor([tgt_vocab[splits[0]]]).long(),
                            source=torch.Tensor(list(float(x) for x in splits[1:])).unsqueeze(dim=0))

        self.softmax = torch.softmax

        triu_backup = torch.ones((1024, 1024)).triu(1).eq(1)
        mask_backup = torch.arange(0, 1024, requires_grad=False).unsqueeze(dim=0).int()

        self.register_buffer(name='triu_backup', tensor=triu_backup)
        self.register_buffer(name='mask_backup', tensor=mask_backup)

        self.update_decay = update_decay

        return

    def init_parameters(self):
        with torch.no_grad():
            self.embeddings.init_parameters()
            self.encoder.init_parameters()
            self.decoder.init_parameters()
        return

    def forward(self, source_enumerate, target_enumerate, src_mask, tgt_mask, crs_mask, ground_truth, eval_variance=False):
        src_embs = self.embeddings.get_src_embs(source_enumerate)
        # context_vector = checkpoint(self.encoder, src_embs, src_mask)
        context_vector = self.encoder(src_embs, src_mask)

        tgt_embs = self.embeddings.get_tgt_embs(target_enumerate)
        # output = checkpoint(self.decoder, tgt_embs, tgt_mask, context_vector, crs_mask)
        output = self.decoder(tgt_embs, tgt_mask, context_vector, crs_mask)
        logits = self.embeddings.get_logits(output)

        if eval_variance:
            # probs for variance calculation
            logits = pad(logits.view(-1, logits.size(-1)), pad=[1, 0])
            new_idxs = torch.arange(start=0, end=logits.size(0), dtype=torch.long, device=logits.device) \
                       * logits.size(1)
            probs = logits.take(new_idxs + ground_truth.view(-1))
            return probs
        else:
            return torch.argmax(logits, dim=-1) + 1, self.criterion(logits, ground_truth)

    def shifted_target(self, target_enumerate: torch.Tensor):
        target_input = target_enumerate[:, :-1]
        target_input = target_input.masked_fill(target_input.eq(self.tgt_eos_idx), self.tgt_pad_idx)
        target_output = target_enumerate[:, 1:]

        return target_input.contiguous(), target_output.contiguous()

    def generate_train_mask(self, source_lengths: torch.Tensor, target_lengths: torch.Tensor,
                            batch_size: int, src_len_max: int, tgt_len_max: int):
        src_mask = self.mask_backup[:, :src_len_max].repeat(batch_size, 1).ge(source_lengths.unsqueeze(dim=1)). \
            unsqueeze(dim=1).repeat(1, src_len_max, 1)
        tgt_mask = self.triu_backup[:tgt_len_max, :tgt_len_max].unsqueeze(dim=0).repeat(batch_size, 1, 1) | \
                   self.mask_backup[:, :tgt_len_max].repeat(batch_size, 1).ge(target_lengths.unsqueeze(dim=1)). \
                       unsqueeze(dim=1).repeat(1, tgt_len_max, 1)
        crs_mask = self.mask_backup[:, :src_len_max].repeat(batch_size, 1).ge(source_lengths.unsqueeze(dim=1)). \
            unsqueeze(dim=1).repeat(1, tgt_len_max, 1)

        return src_mask, tgt_mask, crs_mask

    def generate_loss_mask(self, batch_size: int, target_lengths: torch.Tensor, tgt_len_max: int,
                           backward_factor: float):
        target_lengths = target_lengths.unsqueeze(dim=-1)
        loss_mask = self.mask_backup[:, :tgt_len_max].repeat(batch_size, 1).lt(target_lengths).float().\
            mul(backward_factor).div(target_lengths.float()).view(-1)

        return loss_mask

    def chunk(self, batch):
        src, tgt, src_lens, tgt_lens = batch

        return zip(src.chunk(chunks=self.update_decay, dim=0),
                   tgt.chunk(chunks=self.update_decay, dim=0),
                   src_lens.chunk(chunks=self.update_decay, dim=0),
                   tgt_lens.chunk(chunks=self.update_decay, dim=0))

    def train_step(self, batch, backward_factor):
        all_acc_tokens = 0.0
        all_loss = 0.0

        for s, t, sl, tl in self.chunk(batch):
            t_in, t_out = self.shifted_target(t)
            bs, sl_max = s.size()
            tl_max = int(t_in.size(1))
            sm, tm, cm = self.generate_train_mask(sl, tl - 1, bs, sl_max, tl_max)
            output, loss = self(s, t_in, sm, tm, cm, t_out)
            loss_mask = self.generate_loss_mask(bs, tl - 1, tl_max, backward_factor)
            loss.backward(loss_mask)

            loss_value = float(loss.sum()) / (float(tl.sum()) - bs) * bs
            tgt_mask = t_out.ne(self.tgt_pad_idx)
            acc_value = float(output.masked_select(tgt_mask).eq(t_out.masked_select(tgt_mask)).sum().float())

            all_acc_tokens += acc_value
            all_loss += loss_value

        return all_acc_tokens, all_loss

    def monte_carlo_sample(self, src_input: torch.Tensor, tgt_input: torch.Tensor,
                           src_mask: torch.Tensor, tgt_mask: torch.Tensor, crs_mask: torch.Tensor,
                           tgt_output: torch.Tensor, n_times: int):
        all_probs = list()

        with torch.no_grad():
            for i in range(0, n_times):
                src_embs = self.embeddings.get_src_embs(src_input)
                context_vector = self.encoder(src_embs, src_mask)
                tgt_embs = self.embeddings.get_tgt_embs(tgt_input)
                output = self.decoder(tgt_embs, tgt_mask, context_vector, crs_mask)

                logits = self.embeddings.get_logits(output).softmax(dim=-1)

                logits = pad(logits.view(-1, logits.size(-1)), pad=[1, 0])
                new_idxs = torch.arange(start=0, end=logits.size(0), dtype=torch.long, device=logits.device) \
                           * logits.size(1)
                probs = logits.take(new_idxs + tgt_output.view(-1))
                all_probs.append(probs)

        all_probs = torch.cat(list(x.unsqueeze_(dim=0) for x in all_probs), dim=0)
        return all_probs

    def esti_variance_step(self, batch, n_times=5):

        s, t, sl, tl = batch
        t_in, t_out = self.shifted_target(t)
        bs, sl_max = s.size()
        tl_max = int(t_in.size(1))
        sm, tm, cm = self.generate_train_mask(sl, tl, bs, sl_max, tl_max)

        all_probs = list()
        for i in range(0, n_times):
            probs = self(s, t_in, sm, tm, cm, t_out, eval_variance=True)
            all_probs.append(probs)
        all_probs = torch.cat(list(x.unsqueeze_(dim=0) for x in all_probs), dim=0)

        # calculate token level variance in one batch
        token_variance = all_probs.std(dim=0)
        # use average of all tokens as batch variance
        batch_variance = token_variance.mean()

        batch_variance_cpu = batch_variance.detach().cpu().numpy()
        return batch_variance_cpu

    def eval_step(self, batch):
        s, t, sl, tl = batch
        t_in, t_out = self.shifted_target(t)
        bs, sl_max = s.size()
        tl_max = int(t_in.size(1))
        sm, tm, cm = self.generate_train_mask(sl, tl, bs, sl_max, tl_max)
        output, loss = self(s, t_in, sm, tm, cm, t_out)
        loss_value = float(loss.sum()) / (float(tl.sum()) - bs) * bs
        tgt_mask = t_out.ne(self.tgt_pad_idx)
        acc_value = float(output.masked_select(tgt_mask).eq(t_out.masked_select(tgt_mask)).sum().float()) \
                    / float(tl.sum() - bs) * bs

        return acc_value, loss_value

    def infer_step(self, source_enumerate: torch.Tensor):
        num_of_examples, src_len_max = source_enumerate.size()
        device = self.triu_backup.device
        tgt_sos_token = torch.Tensor([self.tgt_sos_idx]).view((1, 1)) \
            .repeat((num_of_examples * self.beams, 1)).contiguous().long().to(device)

        if self.infer_max_seq_length_mode == 'absolute':
            tgt_infer_max_seq_length = list(self.infer_max_seq_length for _ in range(0, num_of_examples))
        else:
            src_lengths = source_enumerate.ne(self.src_pad_idx).sum(dim=-1).cpu().tolist()
            tgt_infer_max_seq_length = list(x + self.infer_max_seq_length for x in src_lengths)
        beam_search = list(BeamSearch(beams=self.beams,
                                      max_seq_length=l,
                                      tgt_eos_idx=self.tgt_eos_idx,
                                      device=device,
                                      length_penalty=self.length_penalty) for l in tgt_infer_max_seq_length)

        src_mask = source_enumerate.eq(self.src_pad_idx).unsqueeze(dim=1). \
            expand(size=(num_of_examples, src_len_max, src_len_max))
        crs_mask_temp = source_enumerate.eq(self.src_pad_idx).unsqueeze(dim=1)

        src_embs = self.embeddings.get_src_embs(source_enumerate)
        context_vector = self.encoder(src_embs, src_mask)
        next_input = tgt_sos_token
        non_updated_index = list(range(0, num_of_examples))
        non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

        for i in range(0, self.infer_max_seq_length if self.infer_max_seq_length_mode == 'absolute'
                       else self.infer_max_seq_length + src_len_max):
            context_vector_step = context_vector.index_select(dim=0, index=non_updated_index_tensor)
            context_vector_step = torch.cat(list(x.unsqueeze(dim=0).repeat(self.beams, 1, 1)
                                                 for x in context_vector_step.unbind(dim=0)), dim=0)
            tgt_mask = self.triu_backup[:i + 1, :i + 1].unsqueeze(dim=0).expand(
                (len(non_updated_index) * self.beams, i + 1, i + 1))
            crs_mask = crs_mask_temp.index_select(dim=0, index=non_updated_index_tensor).repeat(repeats=(1, i + 1, 1))
            crs_mask = torch.cat(list(x.unsqueeze(dim=0).repeat(self.beams, 1, 1) for x in crs_mask.unbind(dim=0)),
                                 dim=0)
            tgt_embs = self.embeddings.get_tgt_embs(next_input)

            step_output = self.decoder(tgt_embs, tgt_mask, context_vector_step, crs_mask)
            step_output_logits = self.embeddings.get_logits(step_output)
            step_output_softmax = self.softmax(step_output_logits[:, -1:, :], dim=-1)
            step_output_softmax = pad(step_output_softmax, pad=[1, 0], mode='constant', value=0.0)

            for idx, probs in zip(non_updated_index, step_output_softmax.split(self.beams, dim=0)):
                beam_search[idx].routes(probs)

            non_updated_index = list(x[0] for x in filter(
                lambda d: not (d[1].all_eos_updated() or d[1].reach_max_length()), enumerate(beam_search)))
            non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

            if len(non_updated_index) == 0:
                break

            next_input = torch.cat((tgt_sos_token[:len(non_updated_index) * self.beams],
                                    torch.cat(list(beam_search[idx].next_input() for idx in non_updated_index), dim=0)),
                                   dim=-1)

        result = list(beam.get_best_route().tolist() for beam in beam_search)

        return result

    def model_parameters_statistic(self):
        logs = ['Parameters: %d' % sum(p.numel() for p in self.parameters())]

        for name, parameters in self.named_parameters():
            logs.append('%8d\t%20s\t%s' % (parameters.numel(), list(parameters.size()), name))

        logs.append('*' * 80)

        print('\n'.join(logs))
        return '\n'.join(logs)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
