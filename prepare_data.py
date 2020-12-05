import argparse
import os
import random
import shutil
import torchtext.data
import pickle
import onmt.myutils as myutils
from collections import defaultdict

import onmt.inputters as inputters

# ---------------------------------------------------------
# Generate datasets for meta-train, meta-dev and meta-test
# 
# For reptile, the format is:
#     meta-train:
#         task 0: [train]
#         task 1: [train]
#         ...   : [train]
#     meta-dev:
#         task 0: [train, dev, test]
#     meta_test:
#         task 0: [train, dev, test]
#         ...   : [train, dev, test]
# ---------------------------------------------------------

tgt_task_list = \
    [
        'CHEMBL262',
        'CHEMBL267',
        'CHEMBL3267',
        'CHEMBL3650',
        'CHEMBL4005',
        'CHEMBL4282',
        'CHEMBL4722'
    ]


def build_save_vocab(train_dataset, fields, savepath, opt):
    """ Building and saving the vocab """

    fields = inputters.build_vocab(train_dataset, fields, data_type='text',
                                   share_vocab=True,
                                   src_vocab_path='',
                                   src_vocab_size=100,
                                   src_words_min_frequency=1,
                                   tgt_vocab_path='',
                                   tgt_vocab_size=100,
                                   tgt_words_min_frequency=1)
    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = savepath + '/vocab.pt'
    with open(vocab_file,'wb') as f:
        pickle.dump(inputters.save_fields_to_vocab(fields), f)

def build_save_dataset(corpus_type, fields, src_corpus, tgt_corpus, savepath, args):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'dev', 'test']
    dataset = inputters.build_dataset(
        fields, data_type='text',
        src_path=src_corpus,
        tgt_path=tgt_corpus,
        src_dir='',
        src_seq_length=args.max_src_len,
        tgt_seq_length=args.max_tgt_len,
        src_seq_length_trunc=0,
        tgt_seq_length_trunc=0,
        dynamic_dict=True)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    for i in range(len(dataset)):
        if i % 500 == 0:
            print(i)
        setattr(dataset.examples[i], 'graph', myutils.str2graph(dataset.examples[i].src))

    pt_file = "{:s}/{:s}.pt".format(savepath, corpus_type)
    # torch.save(dataset, pt_file)
    with open(pt_file,'wb') as f:
            pickle.dump(dataset, f)
    return [pt_file]

def tokenize(smiles):
    tokens = ' '.join([c for c in smiles])
    patterns = ['B r', 'C l', 'S i', 'S e']
    t_patterns = ['Br', 'Cl', 'Si', 'Se']

    for i in range(len(patterns)):
        while tokens.find(patterns[i]) != -1:
            tokens = tokens.replace(patterns[i], t_patterns[i])
    return tokens

def save_dataset(src_train, tgt_train, dev_mols, test_mols, path):
    src_train_file = open(path + '/src-train.txt', 'w', encoding='utf-8')
    for mol in src_train:
        src_train_file.write(tokenize(mol) + '\n')
    src_train_file.close()
    tgt_train_file = open(path + '/tgt-train.txt', 'w', encoding='utf-8')
    for mol in tgt_train:
        tgt_train_file.write(tokenize(mol) + '\n')
    tgt_train_file.close()
    # dummy pairs
    src_dev = open(path + '/src-dev.txt', 'w', encoding='utf-8')
    tgt_dev = open(path + '/tgt-dev.txt', 'w', encoding='utf-8')
    for mol in dev_mols:
        src_dev.write(tokenize(mol) + '\n')
        tgt_dev.write(tokenize(mol) + '\n')
    tgt_dev.close()
    src_dev.close()
    src_test = open(path + '/src-test.txt', 'w', encoding='utf-8')
    tgt_test = open(path + '/tgt-test.txt', 'w', encoding='utf-8')
    for mol in test_mols:
        src_test.write(tokenize(mol) + '\n')
        tgt_test.write(tokenize(mol) + '\n')
    tgt_test.close()
    src_test.close()
    return

def read_mols(filename):
    mols = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            mol = line.strip()
            mols.append(mol)
    return mols

def read_mol_pairs(filename, evaluate_mols=None):
    # pairs = defaultdict(list)
    srcs, tgts = [], []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            src, tgt = line.strip().split(' ')
            if evaluate_mols != None and (src in evaluate_mols or tgt in evaluate_mols):
                continue 
            # pairs[src].append(tgt)
            srcs.append(src)
            tgts.append(tgt)
    return srcs, tgts

def dump_dataset(savepath, save_dev=False):
    src_corpus = savepath + '/src-train.txt'
    tgt_corpus = savepath + '/tgt-train.txt'

    src_nfeats = inputters.get_num_features('text', src_corpus, 'src')
    tgt_nfeats = inputters.get_num_features('text', tgt_corpus, 'tgt')
    fields = inputters.get_fields('text', src_nfeats, tgt_nfeats)
    fields['graph'] = torchtext.data.Field(sequential = False)
    train_dataset_files = build_save_dataset('train', fields, src_corpus, tgt_corpus, savepath, args)
    
    if save_dev:
        src_corpus = savepath + '/src-dev.txt'
        tgt_corpus = savepath + '/tgt-dev.txt'
        build_save_dataset('dev', fields, src_corpus, tgt_corpus, savepath, args)
    build_save_vocab(train_dataset_files, fields, savepath, args)


def generate_all_train(args, evaluate_mols):
    # gather different pairs.txt
    task_list = os.listdir('data/meta-src')
    all_srcs, all_tgts = [], []
    for task in task_list:
        # write down src-train and tgt-train
        srcs, tgts = read_mol_pairs('data/meta-src/' + task + '/train_pairs.txt', evaluate_mols)
        all_srcs = all_srcs + srcs
        all_tgts = all_tgts + tgts
    savepath = 'processed_data/all-train'
    src_corpus = savepath + '/src-train.txt'
    tgt_corpus = savepath + '/tgt-train.txt'
    
    assert len(all_srcs) == len(all_tgts)
    # remove duplicate mol pairs
    pairs = set()
    for i in range(len(all_srcs)):
        pairs.add(all_srcs[i] + '\t' + all_tgts[i])
    all_srcs, all_tgts = [], []
    for pair in pairs:
        src, tgt = pair.split('\t')
        all_srcs.append(src)
        all_tgts.append(tgt)

    print('All train size: {}'.format(len(all_srcs)))
    src_train = open(src_corpus, 'w', encoding='utf-8')
    for mol in all_srcs:
        src_train.write(tokenize(mol) + '\n')
    src_train.close()

    tgt_train = open(tgt_corpus, 'w', encoding='utf-8')
    for mol in all_tgts:
        tgt_train.write(tokenize(mol) + '\n')
    tgt_train.close()

    # dump data into pt files
    dump_dataset(savepath)
    return

def generate_meta_train(args, evaluate_mols):
    # copy different pairs.txt to new directories
    task_list = os.listdir('data/meta-src')
    for task in task_list:
        savepath = 'processed_data/meta-train/' + task
        os.mkdir(savepath)
        # write down src-train and tgt-train
        src_corpus = savepath + '/src-train.txt'
        tgt_corpus = savepath + '/tgt-train.txt'
        
        srcs, tgts = read_mol_pairs('data/meta-src/' + task + '/train_pairs.txt', evaluate_mols)
        # dev_mols = read_mols('data/meta-src/' + task + '/valid.txt')
        # test_mols = read_mols('data/meta-src/' + task + '/test.txt')

        src_train = open(src_corpus, 'w', encoding='utf-8')
        for mol in srcs:
            src_train.write(tokenize(mol) + '\n')
        src_train.close()

        tgt_train = open(tgt_corpus, 'w', encoding='utf-8')
        for mol in tgts:
            tgt_train.write(tokenize(mol) + '\n')
        tgt_train.close()

        dump_dataset(savepath)
    return

def generate_maml_train(args, evaluate_mols):
    # copy different pairs.txt to new directories
    task_list = os.listdir('data/meta-src')
    for task in task_list:
        savepath = 'processed_data/meta-train/' + task
        os.mkdir(savepath)
        # write down src-train and tgt-train
        src_corpus = savepath + '/src-train.txt'
        tgt_corpus = savepath + '/tgt-train.txt'
        
        srcs, tgts = read_mol_pairs('data/meta-src/' + task + '/pairs.txt', evaluate_mols)
        
        src_list = list(set(srcs))
        random.shuffle(src_list)
        test_size = int(len(src_list) * 0.5)
        test_mols = src_list[:test_size]
        train_mols = src_list[test_size:]

        src_train_mols, tgt_train_mols = [], []
        for i in range(len(srcs)):
            src_train_mols.append(srcs[i])
            tgt_train_mols.append(tgts[i])
        
        src_train, src_dev = src_train_mols[:len(src_train_mols)//2], src_train_mols[len(src_train_mols)//2:]
        tgt_train, tgt_dev = tgt_train_mols[:len(tgt_train_mols)//2], tgt_train_mols[len(tgt_train_mols)//2:]

        src_train_file = open(savepath + '/src-train.txt', 'w', encoding='utf-8')
        for mol in src_train:
            src_train_file.write(tokenize(mol) + '\n')
        src_train_file.close()
        tgt_train_file = open(savepath + '/tgt-train.txt', 'w', encoding='utf-8')
        for mol in tgt_train:
            tgt_train_file.write(tokenize(mol) + '\n')
        tgt_train_file.close()
        # dummy pairs
        src_dev_files = open(savepath + '/src-dev.txt', 'w', encoding='utf-8')
        tgt_dev_files = open(savepath + '/tgt-dev.txt', 'w', encoding='utf-8')
        for mol in src_dev:
            src_dev_files.write(tokenize(mol) + '\n')
        src_dev_files.close()
        for mol in tgt_dev:
            tgt_dev_files.write(tokenize(mol) + '\n')
        tgt_dev_files.close()
        dump_dataset(savepath, save_dev=True)
    return

def generate_meta_dev(args):
    task = args.meta_dev_task
    savepath = 'processed_data/meta-dev/' + task
    os.mkdir(savepath)

    all_mols = set()

    src_train, tgt_train = read_mol_pairs('data/meta-tgt/' + task + '/train_pairs.txt')
    dev_mols = read_mols('data/meta-tgt/' +task + '/valid.txt')
    test_mols = read_mols('data/meta-tgt/' +task + '/test.txt')

    save_dataset(src_train, tgt_train, dev_mols, test_mols, savepath)

    dump_dataset(savepath, save_dev=True)

    for mol in src_train + tgt_train + dev_mols + test_mols:
        all_mols.add(mol)

    return all_mols

def generate_meta_test(args):
    all_mols = set()
    for task in tgt_task_list:
        if task == args.meta_dev_task:
            continue
        savepath = 'processed_data/meta-test/' + task
        os.mkdir(savepath)
        src_train, tgt_train = read_mol_pairs('data/meta-tgt/' + task + '/train_pairs.txt')
        dev_mols = read_mols('data/meta-tgt/' + task + '/valid.txt')
        test_mols = read_mols('data/meta-tgt/' + task + '/test.txt')
        save_dataset(src_train, tgt_train, dev_mols, test_mols, savepath)

        dump_dataset(savepath, save_dev=True)

        for mol in src_train + tgt_train + dev_mols + test_mols:
            all_mols.add(mol)

    return all_mols

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-meta_dev_task', type=str, choices=tgt_task_list, default='CHEMBL4722')
    parser.add_argument('-max_src_len', type=int, default=200)
    parser.add_argument('-max_tgt_len', type=int, default=200)
    
    args = parser.parse_args()

    random.seed(args.seed)

    if os.path.exists('processed_data'):
        # remove everything in processed_data folder
        shutil.rmtree('processed_data/')

    os.mkdir('processed_data')
    os.mkdir('processed_data/meta-train')
    os.mkdir('processed_data/meta-dev')
    os.mkdir('processed_data/meta-test')
    os.mkdir('processed_data/all-train')

    dev_mols = generate_meta_dev(args)
    test_mols = generate_meta_test(args)
    
    evaluate_mols = dev_mols.union(test_mols)

    generate_all_train(args, evaluate_mols)
    generate_meta_train(args, evaluate_mols)
