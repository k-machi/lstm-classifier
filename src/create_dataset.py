from sklearn.model_selection import train_test_split


def get_data(words, labels):
    ret = []
    for word, label in zip(words, labels):
        ret.append(word+'\t'+label)

    return ret


def split_data(words, labels):
    train_X, rest_X, train_y, rest_y = (
                            train_test_split(words, labels, train_size=0.8))
    
    dev_X, test_X, dev_y, test_y = (
                            train_test_split(rest_X, rest_y, train_size=0.5))
    train_data = get_data(train_X, train_y)
    dev_data = get_data(dev_X, dev_y)
    test_data = get_data(test_X, test_y)

    return train_data, dev_data, test_data


def main():

    lines = []
    with open('../data/chemdner_corpus/training.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    with open('../data/chemdner_corpus/development.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    with open('../data/chemdner_corpus/evaluation.annotations.txt', 'r') as f:
        for line in f:
            items=line.strip('\n').split('\t')
            if len(items[4])<1:
                continue
            lines.append(items[4]+'\t'+items[5])

    lines = list(set(lines))

    words = []
    labels = []
    for line in lines:
        items = line.split('\t')
        words.append(items[0])
        labels.append(items[1])

    train, dev, test = split_data(words, labels)

    with open('../data/toy_data/train.tsv', 'w') as f:
        f.write('\n'.join(train))
        print('../data/toy_data/train.tsv')

    with open('../data/toy_data/dev.tsv', 'w') as f:
        f.write('\n'.join(dev))
        print('../data/toy_data/dev.tsv')

    with open('../data/toy_data/test.tsv', 'w') as f:
        f.write('\n'.join(test))
        print('../data/toy_data/test.tsv')


if __name__=='__main__':
    main()
