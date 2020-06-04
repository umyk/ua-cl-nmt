import torch
import re
from model.model import TransformerMT
from corpus.corpus import Corpus
from utils.bleu import BLEU
from optim.label_smoothing import LabelSmoothing

from argparse import ArgumentParser
from copy import deepcopy

argparser = ArgumentParser(description='Transformer')

argparser.add_argument('--prefix', type=str, help='Prefix of all files', default='')

argparser.add_argument('--src_test', type=str, nargs='+', help='Source test file name', required=True)
argparser.add_argument('--tgt_test', type=str, nargs='+', help='Target test file name', required=True)

argparser.add_argument('--src_prefix', type=str, help='Prefix of all source files', default='')
argparser.add_argument('--tgt_prefix', type=str, help='Prefix of all target files', default='')
argparser.add_argument('--src_suffix', type=str, help='Suffix of all source files', default='')
argparser.add_argument('--tgt_suffix', type=str, help='Suffix of all target files', default='')

argparser.add_argument('--src_vocab', type=str, help='Path of source vocabulary', required=True)
argparser.add_argument('--tgt_vocab', type=str, help='Path of target vocabulary', required=True)
argparser.add_argument('--joint_vocab', type=str, help='Path of joint vocabulary', default='')

argparser.add_argument('--model_prefix', type=str, help='Prefix of all model files', default='')
argparser.add_argument('--model_suffix', type=str, help='Suffix of all model files', default='')
argparser.add_argument('--model', type=str, nargs='+', help='Path of storaged model', required=True)
argparser.add_argument('--mode', type=str, choices=['separate', 'average', 'ensemble'],
                       help='Mode of translate for multi models. If multi paths of models are inputted,'
                            'this denotes the mode of using models. "Separate" means using single model recurrently, '
                            '"average" means averaging all input models, and "ensemble" means ensembling all models '
                            'by averaging the probabilities of softmax logits.',
                       default='separate')
argparser.add_argument('--params', type=str, help='Path of storaged parameters', required=True)

argparser.add_argument('--batch_size', type=int, help='Batch size', default=32)
argparser.add_argument('--beam_size', type=int, nargs='+', help='Beam size of beam search', default=1)
argparser.add_argument('--length_penalty', type=float, nargs='+', help='Length penalty alpha when decoding', default=1.0)
argparser.add_argument('--infer_max_seq_length', type=int, help='Max length of sequences when translating', default=256)
argparser.add_argument('--infer_max_seq_length_mode', type=str, choices=['relative', 'absolute'],
                       help='Determine "infer_max_seq_length" is used as absolute length or additive relative length. '
                            'For the latter, sequence length will be the sum of source length and "infer_max_seq_length".',
                       default='absolute')

argparser.add_argument('--output_prefix', type=str, help='Prefix of output files', default='')

argparser.add_argument('--device', type=int, help='device to use', required=True)

main_args = argparser.parse_args()


def translate():
    if len(main_args.src_test) != len(main_args.tgt_test):
        print('Number of source test files %d does not match with target files %d.'
              % (len(main_args.src_test), len(main_args.tgt_test)))
        return

    src_paths = list(main_args.src_prefix + x + main_args.src_suffix for x in main_args.src_test)
    tgt_paths = list(main_args.tgt_prefix + x + main_args.tgt_suffix for x in main_args.tgt_test)

    args = {'file_prefix': '',
            'num_of_layers': '',
            'num_of_heads': '',
            'src_vocab_size': '',
            'tgt_vocab_size': '',
            'embedding_size': '',
            'applied_bpe': '',
            'bpe_suffix_token': '@@',
            'share_embedding': '',
            'share_projection_and_embedding': '',
            'emb_norm_clip': '',
            'emb_norm_clip_type': '',
            'positional_encoding': '',
            'bpe_src': '',
            'bpe_tgt': '',
            'tgt_character_level': '',
            'src_vocab': '',
            'tgt_vocab': '',
            'joint_vocab': '',
            'feedforward_size': '',
            'layer_norm_pre': '',
            'layer_norm_post': '',
            'layer_norm_encoder_start': '',
            'layer_norm_encoder_end': '',
            'layer_norm_decoder_start': '',
            'layer_norm_decoder_end': '',
            'activate_function_name': '',
            'src_pad_token': '',
            'src_unk_token': '',
            'src_sos_token': '',
            'src_eos_token': '',
            'tgt_pad_token': '',
            'tgt_unk_token': '',
            'tgt_eos_token': '',
            'tgt_sos_token': '',
            'optimizer': '',
            'label_smoothing': ''}

    with open(main_args.params, 'r') as f:
        for _, line in enumerate(f):
            splits = line.split()

            if splits[0] in args.keys():
                if len(splits) == 2:
                    args[splits[0]] = splits[1]
                    if args[splits[0]] == 'True':
                        args[splits[0]] = True
                    elif args[splits[0]] == 'False':
                        args[splits[0]] = False
                elif len(splits) == 1:
                    args[splits[0]] = None

    device = torch.device(main_args.device)

    corpus = Corpus(
        prefix=main_args.prefix,
        corpus_source_train='',
        corpus_source_valid='',
        corpus_source_test=src_paths,
        corpus_target_train='',
        corpus_target_valid='',
        corpus_target_test=tgt_paths,
        bpe_suffix_token=args['bpe_suffix_token'],
        bpe_src=args['bpe_src'],
        bpe_tgt=args['bpe_tgt'],
        share_embedding=args['share_embedding'],
        min_seq_length=1,
        max_seq_length=128,
        batch_size=main_args.batch_size,
        length_merging_mantissa_bits=2,
        src_pad_token=args['src_pad_token'],
        src_unk_token=args['src_unk_token'],
        src_sos_token=args['src_sos_token'],
        src_eos_token=args['src_eos_token'],
        tgt_pad_token=args['tgt_pad_token'],
        tgt_unk_token=args['tgt_unk_token'],
        tgt_sos_token=args['tgt_sos_token'],
        tgt_eos_token=args['tgt_eos_token'],
        logger=None,
        num_of_workers=1,
        num_of_steps=1,
        batch_capacity=1024,
        train_buffer_size=1,
        train_prefetch_size=1,
        device=device)
    corpus.build_vocab(src_vocab_size=0, tgt_vocab_size=0,
                       src_vocab_path=main_args.src_vocab,
                       tgt_vocab_path=main_args.tgt_vocab,
                       joint_vocab_path=main_args.joint_vocab)
    corpus.test_file_stats()
    corpus.corpus_numerate_test()

    model = TransformerMT(
        src_vocab_size=corpus.src_vocab_size,
        tgt_vocab_size=corpus.tgt_vocab_size,
        joint_vocab_size=corpus.joint_vocab_size,
        share_embedding=args['share_embedding'],
        share_projection_and_embedding=args['share_projection_and_embedding'],
        src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
        tgt_pad_idx=corpus.tgt_word2idx[corpus.tgt_pad_token],
        tgt_sos_idx=corpus.tgt_word2idx[corpus.tgt_sos_token],
        tgt_eos_idx=corpus.tgt_word2idx[corpus.tgt_eos_token],
        positional_encoding=args['positional_encoding'],
        emb_size=int(args['embedding_size']),
        feed_forward_size=int(args['feedforward_size']),
        num_of_layers=int(args['num_of_layers']),
        num_of_heads=int(args['num_of_heads']),
        train_max_seq_length=128,
        infer_max_seq_length=main_args.infer_max_seq_length,
        infer_max_seq_length_mode=main_args.infer_max_seq_length_mode,
        batch_size=main_args.batch_size,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        feedforward_dropout_prob=0.0,
        residual_dropout_prob=0.0,
        emb_norm_clip=float(args['emb_norm_clip']),
        emb_norm_clip_type=float(args['emb_norm_clip_type']),
        layer_norm_pre=args['layer_norm_pre'],
        layer_norm_post=args['layer_norm_post'],
        layer_norm_encoder_start=args['layer_norm_encoder_start'],
        layer_norm_encoder_end=args['layer_norm_encoder_end'],
        layer_norm_decoder_start=args['layer_norm_decoder_start'],
        layer_norm_decoder_end=args['layer_norm_decoder_end'],
        activate_function_name=args['activate_function_name'],
        prefix=args['file_prefix'],
        pretrained_src_emb='',
        pretrained_tgt_emb='',
        pretrained_src_eos='',
        pretrained_tgt_eos='',
        src_vocab=args['src_vocab'],
        tgt_vocab=args['tgt_vocab'],
        beams=1,
        length_penalty=1.0,
        criterion=LabelSmoothing(vocab_size=corpus.tgt_vocab_size,
                                 padding_idx=0,
                                 confidence=float(args['label_smoothing'])),
        update_decay=1
    ).to(device)

    print(model)
    print('*' * 80)

    bleu = BLEU()

    print('Translate mode: %s' % main_args.mode)

    if main_args.mode == 'separate':
        print('Testing recurrently with models: \n\t%s' % '\n\t'.join(
            list(main_args.model_prefix + model + main_args.model_suffix for model in main_args.model)))
        print('*' * 80)

        with torch.no_grad():
            for model_idx, model_path in enumerate(main_args.model):
                true_model_path = main_args.model_prefix + model_path + main_args.model_suffix
                print('Loading model from %s ... ' % true_model_path, end='')
                checkpoint = torch.load(true_model_path)
                model.load_state_dict(checkpoint['model'])
                print('done.')
                print('*' * 80)

                model.eval()
                call_test(model=model, corpus=corpus, bleu=bleu, model_idx=model_idx,
                          character_level=args['tgt_character_level'])

    elif main_args.mode == 'average':
        print('Testing averaged model from models: \n\t%s' % '\n\t'.join(
            list(main_args.model_prefix + model + main_args.model_suffix for model in main_args.model)))
        print('*' * 80)

        with torch.no_grad():
            true_model_path = main_args.model_prefix + main_args.model[0] + main_args.model_suffix
            print('Loading model from %s ... ' % true_model_path, end='')
            checkpoint = torch.load(true_model_path)
            model.load_state_dict(checkpoint['model'])
            print('done.')
            print('*' * 80)

            model_temp = deepcopy(model)
            for model_path in main_args.model[1:]:
                true_model_path = main_args.model_prefix + model_path + main_args.model_suffix
                print('Loading model from %s ... ' % true_model_path, end='')
                checkpoint = torch.load(true_model_path)
                model_temp.load_state_dict(checkpoint['model'])
                print('done.')
                for p, p_toadd in zip(model.parameters(), model_temp.parameters()):
                    p += p_toadd

            print('Averaging model ... ', end='')
            num_of_models = len(main_args.model)
            for p in model.parameters():
                p /= num_of_models
            print('done.')
            print('*' * 80)

            model.eval()
            call_test(model=model, corpus=corpus, bleu=bleu, model_idx=0, character_level=args['tgt_character_level'])
    else:
        print('Other functions are still under testing ... ')
        return


def call_test(model: TransformerMT, corpus: Corpus, bleu: BLEU, character_level: bool, model_idx: int):
    for num_of_test in corpus.corpus_source_test.keys():
        source, reference, order = corpus.get_test_batches(num_of_test=num_of_test)
        print('*' * 80)
        hypothesis = list()
        num_of_batches = len(source)
        matrix = dict()
        matrix_character = dict()

        if corpus.bpe_tgt:
            reference = list(list(corpus.byte_pair_handler_tgt.subwords2words(list(corpus.tgt_idx2word[x] for x in l))
                                  for l in target_data_set) for target_data_set in reference)
        else:
            reference = list(list(list(corpus.tgt_idx2word[x] for x in l)
                                  for l in target_data_set) for target_data_set in reference)

        for beam_size in main_args.beam_size:
            model.beams = beam_size
            result = dict()
            result_character = dict()

            for length_penalty in main_args.length_penalty:
                model.length_penalty = length_penalty

                if beam_size == 1:
                    print('Beam size equal to 1 means greedy decoding, '
                          'so skip all other settings with different length_penalty values.')
                    if main_args.length_penalty.index(length_penalty) > 0:
                        continue

                for idx, src in enumerate(source):
                    print('\rTranslating batch %d/%d ... ' % (idx + 1, num_of_batches), end='')
                    output = model.infer_step(src)
                    hypothesis += output
                print('done.')

                if corpus.bpe_tgt:
                    hypothesis = list(corpus.byte_pair_handler_tgt.subwords2words(list(corpus.tgt_idx2word[x] for x in l))
                                      for l in hypothesis)
                else:
                    hypothesis = list(list(corpus.tgt_idx2word[x] for x in l) for l in hypothesis)

                bleu_score = bleu.bleu(hypothesis, reference)
                print('BLEU score: %7.4f' % (bleu_score * 100))
                result[length_penalty] = bleu_score

                if character_level:
                    r = re.compile(r'((?:(?:[a-zA-Z0-9])+[\-\+\=!@#\$%\^&\*\(\);\:\'\"\[\]{},\.<>\/\?\|`~]*)+|[^a-zA-Z0-9])')
                    print('')
                    print('For character-level: ')

                    hypothesis_char = list(' '.join(sum(list(r.findall(x) for x in line), list())).split() for line in hypothesis)
                    reference_char = list(list(' '.join(sum(list(r.findall(x) for x in line), list())).split()
                                               for line in gt_ref) for gt_ref in reference)

                    bleu_score = bleu.bleu(hypothesis_char, reference_char)
                    print('BLEU score: %7.4f' % (bleu_score * 100))
                    result_character[length_penalty] = bleu_score

                if main_args.output_prefix == '':
                    print('No output file prefix given, so no output file for storaging candidates ... ')
                else:
                    output_file_path = main_args.output_prefix + '_' + str(model_idx) + '_' + str(num_of_test) + '.txt'
                    print('Output candidates to file: %s' % output_file_path)
                    with open(output_file_path, mode='w', encoding='utf-8') as f:
                        if character_level:
                            hyp_tofile = list(x[1] for x in sorted(zip(order, hypothesis_char), key=lambda d: d[0]))
                        else:
                            hyp_tofile = list(x[1] for x in sorted(zip(order, hypothesis), key=lambda d: d[0]))

                        for hyp_line in hyp_tofile:
                            f.write(' '.join(hyp_line) + '\n')

                print('*' * 80)

                hypothesis.clear()

            matrix[beam_size] = result
            matrix_character[beam_size] = result_character

        print('Performance matrix:')
        print('Horizontal: length penalty; Vertical: beam size')
        print('\t' + '\t'.join('%5.2f' % x for x in main_args.length_penalty))
        for beam_size in matrix.keys():
            print('%2d\t' % beam_size + '\t'.join('%7.4f' % (x * 100) for x in matrix[beam_size].values()))

        if character_level:
            print()
            print('Character level:')
            print('\t' + '\t'.join('%5.2f' % x for x in main_args.length_penalty))
            for beam_size in matrix_character.keys():
                print('%2d\t' % beam_size + '\t'.join('%7.4f' % (x * 100) for x in matrix_character[beam_size].values()))

        print('*' * 80)
        print('*' * 80)

    return


if __name__ == '__main__':
    translate()
