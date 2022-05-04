import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def load_data(data_path, voc_dict):

    max_seq_len = 0
    total_data = []

    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            label, content = int(line[0]), line[1].split(' ')

            seg_items_ids = []
            for word in content:
                if word in voc_dict.keys():
                    seg_items_ids.append(voc_dict[word])
                else:
                    seg_items_ids.append(voc_dict["<UNK>"])

            max_seq_len = max(max_seq_len, len(seg_items_ids))
            total_data.append((label, seg_items_ids))

    return total_data, max_seq_len


def load_dict(dict_path):
    voc_dict = {}
    with open(dict_path, 'r') as fr:
        for line in fr.readlines():
            word, wid = line.strip().split(',')
            voc_dict[word] = int(wid)

    return voc_dict


class WeiboData(Dataset):
    def __init__(self, data_path, dict_path):
        self.voc_dict = load_dict(dict_path)
        self.data, self.max_len = \
            load_data(data_path, self.voc_dict)
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, word_seq = self.data[idx]
        if len(word_seq) < self.max_len:
            word_seq += [self.voc_dict['<PAD>'] for
                         _ in range(self.max_len - len(word_seq))]

        word_seq_padded = torch.LongTensor(word_seq)
        label = torch.tensor(label)

        return word_seq_padded, label


class WeiboData2(Dataset):
    def __init__(self, data_path, dict_path):
        self.voc_dict = load_dict(dict_path)
        self.data, self.max_len = \
            load_data(data_path, self.voc_dict)
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, word_seq = self.data[idx]
        word_seq = torch.LongTensor(word_seq)
        label = torch.tensor(label)

        return word_seq, label

    def collate_fn(self, batch):
        seq_list, tag_list = zip(*batch)
        seq_list = list(seq_list)

        # pad to max length
        seq_one = seq_list[0].numpy()
        seq_one = np.pad(seq_one, (0, self.max_len - len(seq_one)))
        seq_list[0] = torch.from_numpy(seq_one)

        seq_padded = pad_sequence(seq_list, batch_first=True)
        target = torch.tensor(tag_list)

        return seq_padded, target


if __name__ == '__main__':
    data_path = 'data/test_data_0.2.txt'
    dict_path = 'data/voc_dict_0.2.txt'

    dataset = WeiboData2(data_path, dict_path)
    train_loader = DataLoader(dataset,
                              batch_size=100,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=dataset.collate_fn)

    for idx, batch in enumerate(train_loader):
        item_seq, target = batch
        print(item_seq.size())
        print(target.size())

        if idx == 1:
            break





