import os
import random
import jieba
import time


def preprocess(data_ratio=1.,
               test_ratio=0.1,
               train_data_save='train_data_1.txt',
               test_data_save='test_data_1.txt',
               save_vocabulary=False):
    """
    train test split + vocabulary generation

    Parameters
    ----------
    data_ratio: Only use data at a specified ratio. Default: 0.2 .
    test_ratio: Test set ratio. Default: 0.1 .
    train_data_save: train data save path.
    test_data_save: test data save path.
    save_vocabulary: if save vocabulary according to specified data ratio.
    """

    data_path = 'data/meta_data/weibo_senti_100k.csv'
    stopword_path = 'data/meta_data/hit_stopwords.txt'
    train_data_save = os.path.join('data', train_data_save)
    test_data_save = os.path.join('data', test_data_save)

    f_train = open(train_data_save, 'w', encoding='UTF-8')
    f_test = open(test_data_save, 'w', encoding='UTF-8')

    min_seq = 1  # filter short sentence
    top_n = 1000
    UNK = '<UNK>'
    PAD = '<PAD>'

    stop_words = open(stopword_path, 'r', encoding='UTF-8').readlines()
    stop_words = [line.strip() for line in stop_words]
    stop_words.append(' ')

    voc_dict = {}

    print('start processing ...')
    c1 = time.time()
    with open(data_path, 'r', encoding='UTF-8') as f:
        f.readline()
        total_data = f.readlines()
        random.shuffle(total_data)

        data_size = int(len(total_data) * data_ratio)
        test_size = int(data_size * test_ratio)
        train_size = data_size - test_size

        print('test ratio:', test_ratio)
        print(f'train size:{train_size}  test size:{test_size}')

        for idx, line in enumerate(total_data):
            line = line.strip()
            label, content = line[0], line[2:]
            seg_list = jieba.cut(content, cut_all=False)

            seg_items = []
            for word in seg_list:
                if word in stop_words:
                    continue
                seg_items.append(word)
                if not save_vocabulary:
                    continue
                if word in voc_dict.keys():
                    voc_dict[word] = voc_dict[word] + 1
                else:
                    voc_dict[word] = 1

            if idx < test_size:  # save sentence to test data
                f_test.write('{},{}\n'.format(label, ' '.join(seg_items)))
                continue
            f_train.write('{},{}\n'.format(label, ' '.join(seg_items)))

            if idx == data_size - 1:
                break

    f_train.close()
    f_test.close()

    if save_vocabulary:
        print('train test data split successfully')
        voc_dict = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                          key=lambda x: x[1],
                          reverse=True)[: top_n]

        voc_dict = {voc[0]: idx for idx, voc in enumerate(voc_dict)}
        voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})
        print(voc_dict)

        # save vocabulary dictionary
        dict_save = 'data/voc_dict_{}.txt'.format(data_ratio)
        with open(dict_save, 'w') as fw:
            for key, value in voc_dict.items():
                fw.write('{},{}\n'.format(key, value))

        print('save vocabulary dict at data/voc_dict_{}.txt'.format(data_ratio))

    print('total time cost: %.1fs' % (time.time() - c1))


if __name__ == '__main__':
    data_ratio = 1
    train_data = 'train_data_{}.txt'.format(data_ratio)
    test_data = 'test_data_{}.txt'.format(data_ratio)
    preprocess(data_ratio=data_ratio,
               train_data_save=train_data,
               test_data_save=test_data,
               save_vocabulary=True)
