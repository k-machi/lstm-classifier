def main():
    lines = []
    with open('../data/chemdner_corpus/training.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    with open('../data/toy_data/train.tsv', 'w') as f:
        f.write('\n'.join(list(set(lines))))


    lines = []
    with open('../data/chemdner_corpus/development.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    with open('../data/toy_data/dev.tsv', 'w') as f:
        f.write('\n'.join(list(set(lines))))


    lines = []
    with open('../data/chemdner_corpus/evaluation.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    with open('../data/toy_data/test.tsv', 'w') as f:
        f.write('\n'.join(list(set(lines))))


if __name__=='__main__':
    main()
