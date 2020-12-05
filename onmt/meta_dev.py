#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch
import numpy as np
import onmt.opts as opts
import pickle
import torchtext.data
import codecs
import pdb
import gc
from copy import deepcopy

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features, load_fields_from_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
import onmt.decoders.ensemble
from onmt.translate.translator import Translator
from onmt.myutils import read_score_csv, calculate_metrics


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == 'text' and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and
                                 decoder rnn sizes for translation now.""")

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    out_file = None
    best_test_score, best_ckpt = -10000, None
    dummy_parser = argparse.ArgumentParser(description='meta_dev.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    # for i in range(28, opt.meta_iterations * opt.inner_iterations + 28, 28):
    for i in range(57, 57*500 + 57, 57*10):
        ckpt_path = '{}_epoch_{}.pt'.format(opt.save_model, i)
        logger.info('Loading checkpoint from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        fields = load_fields_from_vocab(checkpoint['vocab'], data_type="text")

        # Build model.
        model = build_model(model_opt, opt, fields, checkpoint)
        
        assert opt.train_from == ''  # do not load optimizer state
        optim = build_optim(model, opt, checkpoint)  
        # Build model saver, no need to create task dir for dev
        if not os.path.exists('experiments/meta_dev'):
            os.mkdir('experiments/meta_dev')
            os.mkdir('experiments/meta_dev/' + opt.meta_dev_task)
        elif not os.path.exists('experiments/meta_dev/' + opt.meta_dev_task):
            os.mkdir('experiments/meta_dev/' + opt.meta_dev_task)
        model_saver = build_model_saver(model_opt, 'experiments/meta_dev/' + opt.meta_dev_task + '/model', opt, model, fields, optim)

        trainer = build_trainer(opt, device_id, model, fields, optim, "text", model_saver=model_saver)

        train_iter = list(build_dataset_iter(lazily_load_dataset("train", opt), fields, opt))
        # do training on trainset of meta-dev task
        trainer.train(train_iter, opt.inner_iterations)
        
        # do evaluation on devset of meta-dev task
        best_dev_score, best_model_path = -10000, None
        for model_path in os.listdir('experiments/meta_dev/' + opt.meta_dev_task):
            if model_path.find('.pt') == -1:
                continue
            if out_file is None:
                out_file = codecs.open(opt.output, 'w+', 'utf-8')
            
            fields, model, model_opt = onmt.model_builder.load_test_model(opt, dummy_opt.__dict__, model_path='experiments/meta_dev/' + opt.meta_dev_task + '/' + model_path)

            scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                                    opt.beta,
                                                    opt.coverage_penalty,
                                                    opt.length_penalty)

            kwargs = {k: getattr(opt, k)
                    for k in ["beam_size", "n_best", "max_length", "min_length",
                                "stepwise_penalty", "block_ngram_repeat",
                                "ignore_when_blocking", "dump_beam", "report_bleu",
                                "replace_unk", "gpu", "verbose", "fast", "mask_from"]}
            fields['graph'] = torchtext.data.Field(sequential = False)
            translator = Translator(model, fields, global_scorer=scorer,
                                    out_file=out_file, report_score=False,
                                    copy_attn=model_opt.copy_attn, logger=logger,
                                    log_probs_out_file=None,
                                    **kwargs)
            # make translation and save result
            all_scores, all_predictions = translator.translate(
                src_path='processed_data/meta-dev/' + opt.meta_dev_task + '/src-dev.txt',
                tgt_path=None,
                src_dir=None,
                batch_size=opt.translate_batch_size,
                attn_debug=False)
            # dump predictions
            f = open('experiments/meta_dev/' + opt.meta_dev_task + '/dev_predictions.csv', 'w', encoding='utf-8')
            f.write('smiles,property\n')
            for n_best_mols in all_predictions:
                for mol in n_best_mols:
                    f.write(mol.replace(' ', '')+',0\n')
            f.close()

            # call chemprop to get scores
            test_path = '\"' + 'experiments/meta_dev/' + opt.meta_dev_task + '/dev_predictions.csv' + '\"'
            checkpoint_path =  '\"' + 'scorer_ckpts/' + opt.meta_dev_task + '/model.pt' + '\"'
            preds_path = '\"' + 'experiments/meta_dev/' + opt.meta_dev_task + '/dev_scores.csv' + '\"'
            
            # in case of all mols are invalid (will produce not output file by chemprop)
            # the predictions are copied into score file
            cmd = 'cp {} {}'.format(test_path, preds_path)
            result = os.popen(cmd)
            result.close()

            # chempro predict score for each mol and save into score file
            cmd = 'python chemprop/predict.py --test_path {} --checkpoint_path {} --preds_path {} --num_workers 0'.format(test_path, checkpoint_path, preds_path)
            scorer_result = os.popen(cmd)
            scorer_result.close()
            # read score file and get score
            score = read_score_csv('experiments/meta_dev/' + opt.meta_dev_task + '/dev_scores.csv')
            assert len(score) % opt.beam_size == 0
            # dev_scores = []
            # for i in range(0, len(score), opt.beam_size):
            #     dev_scores.append(sum([x[1] for x in score[i:i+opt.beam_size]]) / opt.beam_size)

            # report dev score
            dev_metrics = calculate_metrics(opt.meta_dev_task, 'dev', 'dev', score)
            logger.info('dev metrics: ' + str(dev_metrics))
            dev_score = dev_metrics['success_rate']
            if dev_score > best_dev_score:
                logger.info('New best dev success rate: {:.4f} by {}'.format(dev_score, model_path))
                best_model_path = model_path
                best_dev_score = dev_score
            else:
                logger.info('dev success rate: {:.4f} by {}'.format(dev_score, model_path))
            
            del fields
            del model
            del model_opt
            del scorer
            del translator
            gc.collect()

        assert best_model_path != None
        # do testing on testset of meta-dev task
        if out_file is None:
            out_file = codecs.open(opt.output, 'w+', 'utf-8')
        fields, model, model_opt = onmt.model_builder.load_test_model(opt, dummy_opt.__dict__, model_path='experiments/meta_dev/' + opt.meta_dev_task + '/' + best_model_path)

        scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                                opt.beta,
                                                opt.coverage_penalty,
                                                opt.length_penalty)

        kwargs = {k: getattr(opt, k)
                    for k in ["beam_size", "n_best", "max_length", "min_length",
                                "stepwise_penalty", "block_ngram_repeat",
                                "ignore_when_blocking", "dump_beam", "report_bleu",
                                "replace_unk", "gpu", "verbose", "fast", "mask_from"]}
        fields['graph'] = torchtext.data.Field(sequential = False)
        translator = Translator(model, fields, global_scorer=scorer,
                                out_file=out_file, report_score=False,
                                copy_attn=model_opt.copy_attn, logger=logger,
                                log_probs_out_file=None,
                                **kwargs)
        # make translation and save result
        all_scores, all_predictions = translator.translate(
            src_path='processed_data/meta-dev/' + opt.meta_dev_task + '/src-test.txt',
            tgt_path=None,
            src_dir=None,
            batch_size=opt.translate_batch_size,
            attn_debug=False)
        # dump predictions
        f = open('experiments/meta_dev/' + opt.meta_dev_task + '/test_predictions.csv', 'w', encoding='utf-8')
        f.write('smiles,property\n')
        for n_best_mols in all_predictions:
            for mol in n_best_mols:
                f.write(mol.replace(' ', '')+',0\n')
        f.close()
        # call chemprop to get scores
        test_path = '\"' + 'experiments/meta_dev/' + opt.meta_dev_task + '/test_predictions.csv' + '\"'
        checkpoint_path = '\"' + 'scorer_ckpts/' + opt.meta_dev_task + '/model.pt' + '\"'
        preds_path = '\"' + 'experiments/meta_dev/' + opt.meta_dev_task + '/test_scores.csv' + '\"'
        
        # in case of all mols are invalid (will produce not output file by chemprop)
        # the predictions are copied into score file
        cmd = 'cp {} {}'.format(test_path, preds_path)
        result = os.popen(cmd)
        result.close()

        cmd = 'python chemprop/predict.py --test_path {} --checkpoint_path {} --preds_path {} --num_workers 0'.format(test_path, checkpoint_path, preds_path)
        scorer_result = os.popen(cmd)
        # logger.info('{}'.format('\n'.join(scorer_result.readlines())))
        scorer_result.close()
        # read score file and get score

        score = read_score_csv('experiments/meta_dev/' + opt.meta_dev_task + '/test_scores.csv')
        
        assert len(score) % opt.beam_size == 0
        # test_scores = []
        # for i in range(0, len(score), opt.beam_size):
        #     test_scores.append(sum([x[1] for x in score[i:i+opt.beam_size]]) / opt.beam_size)

        # report if it is the best on test
        test_metrics = calculate_metrics(opt.meta_dev_task, 'dev', 'test', score)
        logger.info('test metrics: ' + str(test_metrics))
        test_score = test_metrics['success_rate']
        if test_score > best_test_score:
            best_ckpt = ckpt_path
            logger.info('New best test success rate: {:.4f} by {}'.format(test_score, ckpt_path))
            best_test_score = test_score
        else:
            logger.info('test success rate: {:.4f} by {}'.format(test_score, ckpt_path))

        del model_opt
        del fields
        del checkpoint
        del model
        del optim
        del model_saver
        del trainer
        gc.collect()
    # if opt.tensorboard:
    #     trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='meta_dev.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
