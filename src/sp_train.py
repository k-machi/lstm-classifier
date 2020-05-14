import os
import argparse
import sentencepiece as spm

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../sp_model/',
                                            help='path to save trained model')
    parser.add_argument('--train_data', type=str,
                            default='../data/toydata/train.tsv',
                                    help='training file of sp-model')
    parser.add_argument('--vocab_size', type=int, default=1000,
                                        help='vocabulary size of sentencepiece')
    args = parser.parse_args()

    return args


def main():
    args = parse_arg()
    model_path = args.model_path
    vocab_size = args.vocab_size
    os.makedirs(model_path, exist_ok=True)

    model_name = model_path+'sp'

    # train_file(tsv): [word, label]
    tsv_file = args.train_data
    with open(tsv_file, 'r') as f:
        words = []
        for line in f.readlines():
            words.append(line.split('\t')[0])

    train_file = model_path+'words.txt'
    with open(train_file, 'w') as f:
        f.write('\n'.join(words))

    spm.SentencePieceTrainer.Train('input={}, \
                                    --model_prefix={} \
                                    --character_coverage=1.0 \
                                    --vocab_size={} \
                                    --pad_id=0 \
                                    --unk_id=1 \
                                    --bos_id=-1 \
                                    --eos_id=-1'
                                    .format(train_file, model_name, vocab_size))

if __name__=='__main__':
    main()
