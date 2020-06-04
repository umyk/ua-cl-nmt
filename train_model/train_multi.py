import torch
from torch import nn
import numpy
import time
import os
import re
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, nvmlDeviceGetMemoryInfo, \
    nvmlSystemGetDriverVersion, nvmlDeviceGetName
from torch.multiprocessing import Pool, Queue
from threading import Thread
from torch.nn.parallel import replicate, scatter, gather
from copy import deepcopy

from model.model import TransformerMT
from corpus.corpus import Corpus,_batch_prefetch, _data_segment
from utils.stats import Stats
from utils.bleu import BLEU
from utils.earlystopping import EarlyStopping, VarianceScorer


class TrainerMultiDevice:
    def __init__(self,
                 model: TransformerMT,
                 corpus: Corpus,
                 optimizer: torch.optim.Optimizer,
                 stats: Stats,
                 bleu: BLEU,
                 tgt_character_level: bool,
                 buffer_every_steps: int,
                 report_every_steps: int,
                 eval_every_steps: int,
                 num_of_steps: int,
                 eval_type: str,
                 processed_steps: int,
                 learning_rate_schedule: str,
                 update_decay: int,
                 batch_capacity: int,
                 max_save_models: int,
                 grad_norm_clip: float,
                 grad_norm_clip_type: float,
                 annotate: str,
                 device_idxs: [int],
                 gpu_memory_limit: float,
                 ):
        self.model = model
        self.corpus = corpus
        self.optimizer = optimizer
        self.stats = stats
        self.bleu = bleu
        self.tgt_character_level = tgt_character_level

        self.buffer_every_steps = buffer_every_steps
        self.report_every_steps = report_every_steps
        self.eval_every_steps = eval_every_steps
        self.num_of_steps = num_of_steps

        self.eval_type = eval_type
        self.processed_steps = processed_steps
        self.update_decay = update_decay
        self.batch_capacity = batch_capacity

        self.src_pad_idx = self.model.src_pad_idx
        self.tgt_eos_idx = self.model.tgt_eos_idx
        self.tgt_pad_idx = self.model.tgt_pad_idx

        self.max_save_models = max_save_models
        self.grad_norm_clip = grad_norm_clip if grad_norm_clip > 0.0 else None
        self.grad_norm_clip_type = grad_norm_clip_type

        self.annotate = annotate

        self.device_idxs = device_idxs
        self.num_of_devices = len(self.device_idxs)
        self.gpu_memory_limit = gpu_memory_limit

        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.best_bleu = 0.0
        self.best_step = 0

        self.lr_schedule = eval(learning_rate_schedule)
        self.lr = 0.005
        self.backward_factor = list()

        self.loss_report = numpy.zeros(self.report_every_steps, dtype=float)
        self.acc_report = numpy.zeros(self.report_every_steps, dtype=float)
        self.update_decay_steps = numpy.zeros(self.report_every_steps, dtype=int)
        self.src_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.tgt_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.src_num_pad_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.tgt_num_pad_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.num_examples = numpy.zeros(self.report_every_steps, dtype=int)
        self.time_sum = 0.0

        self.memory_unit = float(2 ** 30)

        # for uncertainty estimation
        self.esti_variance_every_steps = 1000
        self.tolerance = 4

        nvmlInit()
        print('Driver version: %s' % nvmlSystemGetDriverVersion().decode('utf-8'))

        device_true_idxs = list(int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        self.gpu_info_handler = list(nvmlDeviceGetHandleByIndex(x) for x in device_true_idxs)
        for idx, handler in enumerate(self.gpu_info_handler):
            print('Device no.%d, true idx: %d' % (idx, device_true_idxs[idx]))
            print('\tGPU Name: %s' % nvmlDeviceGetName(handler).decode('utf-8'))

        self.queue = Queue(maxsize=self.num_of_devices)
        self.replicas = list()

        self.async_update_rules = list()
        device_idxs_rules = self.device_idxs.copy()
        while len(device_idxs_rules) > 1:
            rules = dict()
            for i in range(1, len(device_idxs_rules), 2):
                rules[device_idxs_rules[i]] = device_idxs_rules[i - 1]
            device_idxs_rules = device_idxs_rules[::2]
            self.async_update_rules.append(rules)

        return

    def retrain_model(self,
                      retrain_model: str,
                      processed_steps: int):
        self.processed_steps = processed_steps
        self.corpus.num_of_made_batches.value = processed_steps
        self.corpus.num_of_trained_batches.value = processed_steps
        self.corpus.invoke_train_segments_making()
        self.corpus.invoke_train_batches_making()

        with torch.no_grad():
            print('Loading saved model from %s at step %d ... ' % (retrain_model, processed_steps), end='')
            checkpoint = torch.load(retrain_model)

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            torch.cuda.set_rng_state(checkpoint['gpu_random_state'])

            print('done.')

            self.parallel_model()
            print('*' * 80)
            print('Evaluating')
            torch.cuda.empty_cache()
            for model in self.replicas:
                model.eval()
            self.eval_step()
            self.save()
            torch.cuda.empty_cache()
            for model in self.replicas:
                model.train()
            print('Training')
            print(self.annotate)

        return

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'gpu_random_state': torch.cuda.get_rng_state()
        }, self.stats.fold_name + '/' + str(self.processed_steps) + '.pt')

        print('Save to:', self.stats.fold_name + '/' + str(self.processed_steps) + '.pt')
        print('*' * 80)

        return

    def parallel_model(self):
        self.replicas.append(self.model)
        for i in range(1, self.num_of_devices):
            model = deepcopy(self.model).to(self.device_idxs[i])
            self.replicas.append(model)

        return

    def upgrade_competence(self):
        """ upgrade competence based on current model uncertainty """
        self.corpus.model_competence_lv = self.corpus.model_competence_lv + 1
        self.corpus.train_pool[:] = []
        self.corpus.num_of_made_segments.value = 0
        print('Clear training data pool ')

    def run(self):

        while self.processed_steps < self.num_of_steps:

            self.esti_variance_every_steps = _data_segment(self.corpus)
            print('*'*80)

            need_reload = False
            self.corpus.invoke_train_batches_making()

            variance_earlystopper = EarlyStopping(
                tolerance=self.tolerance, scorers=[VarianceScorer()]
            )

            if len(self.replicas) < self.num_of_devices:
                self.replicas.clear()
                self.parallel_model()

            while self.processed_steps < self.num_of_steps and not need_reload:
                next_batches = self.corpus.get_train_batches(self.buffer_every_steps)

                for batch in next_batches:
                    time_start = time.time()

                    self.train_step(batch)          # worker training
                    self.update()                   # worker collection and sync

                    self.time_sum += time.time() - time_start

                    ' estimate variance begin '
                    if self.processed_steps % self.esti_variance_every_steps == 0:

                        with torch.no_grad():
                            print('*' * 80)
                            print('Variance Estimating...')
                            torch.cuda.empty_cache()
                            variance = self.esti_variance_step()
                            torch.cuda.empty_cache()

                            # Run variance converge computer (use patience mechanism)
                            variance_earlystopper(variance, self.processed_steps)
                            # If the patience has reached the limit, upgrade the model competence level
                            if variance_earlystopper.has_stopped():
                                self.upgrade_competence()
                                need_reload = True
                                break

                            print('Training')
                            print(self.annotate)
                    ' estimate variance end '

                    if self.processed_steps % self.report_every_steps == 0:
                        self.report()

                    if self.processed_steps % self.eval_every_steps == 0:
                        with torch.no_grad():
                            print('*' * 80)
                            print('Evaluating')
                            torch.cuda.empty_cache()
                            for model in self.replicas:
                                model.eval()
                            self.eval_step()
                            self.save()
                            torch.cuda.empty_cache()
                            for model in self.replicas:
                                model.train()
                            print('Training')
                            print(self.annotate)

                    if self.processed_steps >= self.num_of_steps:
                        print('End of train.')
                        return

        return

    def report(self):
        self.loss_report /= self.update_decay_steps
        self.acc_report /= self.update_decay_steps

        for acc_step, loss_step in zip(self.acc_report.tolist(), self.loss_report.tolist()):
            self.stats.train_record(acc_step, loss_step)

        infos = list(nvmlDeviceGetMemoryInfo(handler) for handler in self.gpu_info_handler)
        tems = list(nvmlDeviceGetTemperature(handler, 0) for handler in self.gpu_info_handler)
        info_used = sum(info.used for info in infos)
        info_total = sum(info.total for info in infos)

        output_str = str.format(
            'Step: %6d, acc:%6.2f (%6.2f~%6.2f), loss:%5.2f (%5.2f~%5.2f), '
            'lr: %.4f, bc: %d/%d, bs: %5d, tks: %6d+%6d, t: %5.2f, m: %5.2f/%5.2f, tem: %sC'
            % (self.processed_steps,
               self.acc_report.mean() * 100,
               self.acc_report.min() * 100,
               self.acc_report.max() * 100,
               self.loss_report.mean(),
               self.loss_report.min(),
               self.loss_report.max(),
               self.lr,
               self.src_tokens.sum() + self.tgt_tokens.sum(),
               self.src_num_pad_tokens.sum() + self.tgt_num_pad_tokens.sum(),
               self.num_examples.sum(),
               self.src_tokens.sum(),
               self.tgt_tokens.sum(),
               self.time_sum,
               info_used / self.memory_unit,
               info_total / self.memory_unit,
               '/'.join(str(x) for x in tems))
        )

        self.stats.log_to_file(output_str)
        print(output_str)

        self.acc_report.fill(0)
        self.loss_report.fill(0)
        self.update_decay_steps.fill(0)
        self.src_tokens.fill(0)
        self.tgt_tokens.fill(0)
        self.src_num_pad_tokens.fill(0)
        self.tgt_num_pad_tokens.fill(0)
        self.num_examples.fill(0)
        self.time_sum = 0.0

        if max(info.used / info.total for info in infos) > self.gpu_memory_limit:
            torch.cuda.empty_cache()

        return

    def train_step(self, batch):
        report_idx = self.processed_steps % self.report_every_steps

        self.src_num_pad_tokens[report_idx] = int(batch[0].numel())
        self.tgt_num_pad_tokens[report_idx] = int(batch[1].numel())
        self.src_tokens[report_idx] = int(batch[2].sum())
        self.tgt_tokens[report_idx] = int(batch[3].sum())
        self.num_examples[report_idx] = int(batch[0].size(0))

        all_inputs = scatter(inputs=batch, target_gpus=self.device_idxs, dim=0)
        num_of_device = len(all_inputs)
        factor = 1.0 / self.num_examples[report_idx]
        args = list((self.replicas[i], all_inputs[i], factor, self.queue) for i in range(0, num_of_device))

        all_threads = list(Thread(target=_train_worker, args=list(args[i])) for i in range(0, num_of_device))
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        all_results = list(self.queue.get() for _ in range(0, num_of_device))

        self.acc_report[report_idx] += sum(x[0] for x in all_results) / self.tgt_tokens[report_idx]
        self.loss_report[report_idx] += sum(x[1] for x in all_results) / self.num_examples[report_idx]
        self.update_decay_steps[report_idx] += 1

        return

    def update(self):
        for rule_set in self.async_update_rules:
            all_threads = list(Thread(target=_gather_worker, args=(self.replicas[d[1]], self.replicas[d[0]], d[1]))
                               for d in rule_set.items())
            for t in all_threads:
                t.start()
            for t in all_threads:
                t.join()

        if self.grad_norm_clip:
            nn.utils.clip_grad_norm_(self.replicas[0].parameters(),
                                     max_norm=self.grad_norm_clip,
                                     norm_type=self.grad_norm_clip_type)

        self.processed_steps += 1
        self.lr = self.lr_schedule(self.processed_steps)
        self.optimizer.step(self.lr)
        self.optimizer.zero_grad()

        all_threads = list(Thread(target=_asynchronized_worker, args=(self.replicas[0], self.replicas[i], i))
                           for i in self.device_idxs[1:])
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        return

    def esti_variance_step(self):
        source_data, target_data, src_lens, tgt_lens = self.corpus.get_esti_batches()
        whole_batchs_variance_list = []

        for source, target, src_len, tgt_len in zip(source_data, target_data, src_lens, tgt_lens):
            for i in range(0, self.corpus.num_of_multi_refs):
                all_inputs = scatter(inputs=(source, target[i], src_len, tgt_len[i]), target_gpus=self.device_idxs,
                                     dim=0)
                num_of_device = len(all_inputs)

                args = list((self.replicas[i], all_inputs[i]) for i in range(0, num_of_device))

                all_threads = list(Thread(target=_esti_variance_worker, args=list(args[i]) + [self.queue]) for i in
                                   range(0, num_of_device))
                for t in all_threads:
                    t.start()
                for t in all_threads:
                    t.join()

                all_results = list(self.queue.get() for _ in range(0, num_of_device))
                cur_batch_variance = numpy.mean(all_results)
                whole_batchs_variance_list.append(cur_batch_variance)

        total_variance = torch.FloatTensor(whole_batchs_variance_list)
        total_average_variance = total_variance.mean()

        return total_average_variance

    def eval_step(self):
        acc = numpy.zeros(self.corpus.num_of_multi_refs)
        loss = numpy.zeros(self.corpus.num_of_multi_refs)

        source_data, target_data, src_lens, tgt_lens = self.corpus.get_valid_batches()
        source_data_translation, target_data_translation = \
            self.corpus.get_valid_batches_for_translation()

        num_of_batches = len(source_data_translation)
        num_of_examples = len(self.corpus.corpus_source_valid_numerate)

        for source, target, src_len, tgt_len in zip(source_data, target_data, src_lens, tgt_lens):

            for i in range(0, self.corpus.num_of_multi_refs):
                all_inputs = scatter(inputs=(source, target[i], src_len, tgt_len[i]), target_gpus=self.device_idxs,
                                     dim=0)
                num_of_device = len(all_inputs)
                args = list((self.replicas[i], all_inputs[i]) for i in range(0, num_of_device))

                all_threads = list(Thread(target=_eval_worker, args=list(args[i]) + [self.queue]) for i in
                                   range(0, num_of_device))
                for t in all_threads:
                    t.start()
                for t in all_threads:
                    t.join()

                all_results = list(self.queue.get() for _ in range(0, num_of_device))

                acc[i] += sum(x[0] for x in all_results)
                loss[i] += sum(x[1] for x in all_results)

        acc /= num_of_examples
        loss /= num_of_examples

        translation_results = []
        device_idx = self.device_idxs[self.processed_steps // self.eval_every_steps % self.num_of_devices]
        model = self.replicas[device_idx]

        for idx, source in enumerate(source_data_translation):
            print('\rTranslating batch %d/%d ... ' % (idx + 1, num_of_batches), sep=' ', end='')
            translated = model.infer_step(source.to(device_idx))
            translation_results += translated
        print('done.')
        translation_results = list(list(self.corpus.tgt_idx2word[x] for x in line)
                                   for line in translation_results)
        target_data_translation = list(list(list(self.corpus.tgt_idx2word[x] for x in line)
                                            for line in gt_ref)
                                       for gt_ref in target_data_translation)

        if self.corpus.bpe_tgt:
            translation_results = list(self.corpus.byte_pair_handler_tgt.subwords2words(line)
                                       for line in translation_results)
            target_data_translation = list(list(self.corpus.byte_pair_handler_tgt.subwords2words(line)
                                                for line in gt_ref)
                                           for gt_ref in target_data_translation)

        bleu_score = self.bleu.bleu(translation_results, target_data_translation)

        print('BLEU score: %5.2f' % (bleu_score * 100))

        if self.tgt_character_level:
            r = re.compile(r'((?:(?:[a-zA-Z0-9])+[\-\+\=!@#\$%\^&\*\(\);\:\'\"\[\]{},\.<>\/\?\|`~]*)+|[^a-zA-Z0-9])')
            print('')
            print('For character-level:')

            translation_results = list(' '.join(sum(list(r.findall(x) for x in line), list())).split() for line in translation_results)
            target_data_translation = list(list(' '.join(sum(list(r.findall(x) for x in line), list())).split()
                                           for line in gt_ref) for gt_ref in target_data_translation)
            bleu_score = self.bleu.bleu(translation_results, target_data_translation)
            print('BLEU score: %5.2f' % (bleu_score * 100))

        self.stats.valid_record(acc, loss, bleu_score)

        del source_data, target_data, src_lens, tgt_lens, source_data_translation, target_data_translation

        for i in range(0, self.corpus.num_of_multi_refs):
            output = str.format('Step %6d valid, ref%1d acc: %5.2f loss: %5.2f bleu: %f'
                                % (self.processed_steps, i, acc[i] * 100, loss[i], bleu_score))
            print(output)
            self.stats.log_to_file(output)

        print('*' * 80)
        self.stats.log_to_file('*' * 80)
        print('Model performances (%s): ' % self.eval_type)

        if self.eval_type == 'acc':
            sorted_results = sorted(self.stats.valid_acc.items(), key=lambda d: d[1], reverse=True)
            temp_acc = float(acc.mean())
            if self.best_acc < temp_acc:
                print('Best acc: %f -> %f at step %d -> %d' % (self.best_acc, temp_acc,
                                                               self.best_step, self.processed_steps))
                self.best_acc = temp_acc
                self.best_step = self.processed_steps
            else:
                print('Best acc: %f at step %d' % (self.best_acc, self.best_step))

        elif self.eval_type == 'xent':
            sorted_results = sorted(self.stats.valid_loss.items(), key=lambda d: d[1])
            temp_loss = float(loss.mean())
            if self.best_loss > temp_loss:
                print('Best loss: %f -> %f at step %d -> %d' % (self.best_loss, temp_loss,
                                                                self.best_step, self.processed_steps))
                self.best_loss = temp_loss
                self.best_step = self.processed_steps
            else:
                print('Best loss: %f at step %d' % (self.best_loss, self.best_step))

        else:
            sorted_results = sorted(self.stats.valid_bleu.items(), key=lambda d: d[1], reverse=True)
            temp_bleu = float(bleu_score)
            if self.best_bleu < temp_bleu:
                print('Best bleu: %f -> %f at step %d -> %d' % (self.best_bleu, temp_bleu,
                                                                self.best_step, self.processed_steps))
                self.best_bleu = temp_bleu
                self.best_step = self.processed_steps
            else:
                print('Best bleu: %f at step %d' % (self.best_bleu, self.best_step))

        if self.max_save_models > 0:
            for (step_temp, value_temp) in sorted_results[:self.max_save_models]:
                print('%6d\t%8f' % (step_temp, value_temp))

            for (step_temp, _) in sorted_results[self.max_save_models:]:
                path = self.stats.fold_name + '/' + str(step_temp) + '.pt'
                if os.path.isfile(self.stats.fold_name + '/' + str(step_temp) + '.pt'):
                    os.remove(path)
                    print('Remove %d.pt' % step_temp)

        print('*' * 80)

        return


def _train_worker(model, inputs, backward_factor, queue):
    queue.put(model.train_step(inputs, backward_factor))


def _gather_worker(origin_model, replicated_model, origin_device):
    for p_o, p_r in zip(origin_model.parameters(), replicated_model.parameters()):
        p_o.grad += p_r.grad.to(origin_device)


def _asynchronized_worker(origin_model, replicated_model, device_idx):
    replicated_model.zero_grad()
    for p_o, p_r in zip(origin_model.parameters(), replicated_model.parameters()):
        p_r.data.copy_(p_o.data.to(device_idx))


def _eval_worker(model, inputs, queue):
    queue.put(model.eval_step(inputs))


def _esti_variance_worker(model, inputs, queue):
    queue.put(model.esti_variance_step(inputs))


def _infer_worker(model, inputs, queue, idx):
    queue.put(model.infer_step(inputs), idx)
