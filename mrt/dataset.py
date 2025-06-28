import numpy as np


def get_padding(sentences, max_len):
    lengths = np.array([len(s) for s in sentences])
    lengths[lengths > max_len] = max_len
    padded = np.zeros((len(sentences), max_len), dtype=np.int64)
    for i, s in enumerate(sentences):
        pad = np.asarray(s[:lengths[i]])
        padded[i, :lengths[i]] = pad
    return padded


def get_mask_matrix(seq_lengths, max_len):
    mask_matrix = np.zeros((len(seq_lengths), max_len), dtype=np.int64)
    for i, seq_len in enumerate(seq_lengths):
        mask_matrix[i, :min(seq_len, max_len)] = np.ones((min(seq_len, max_len),), dtype=np.int64)
    return mask_matrix


class YDataset(object):
    def __init__(self, processed_data, clone_pair, to_pad=True, cut_len=40):
        self.processed_data = processed_data
        self.clone_pair = clone_pair
        self.to_pad = to_pad
        self.cut_len = cut_len

        self._num_examples = len(self.clone_pair)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def __len__(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _shuffle(self, seed):
        np.random.shuffle(self.clone_pair)

    def all_file_data(self):
        index_1 = []
        file_1 = []
        pre_1 = []
        post_1 = []
        seq_lens = []
        for index in self.processed_data.keys():
            for file in self.processed_data[index].keys():
                file_data = self.processed_data[index][file]
                index_1.append(index)
                file_1.append(file)
                pre_1.append(file_data["preorder"])
                post_1.append(file_data["postorder"])
                seq_lens.append(int(file_data["len"]))

        # max_len1 = min(max(seq_lens[0]), self.cut_len)
        # max_len2 = min(max(seq_lens[1]), self.cut_len)
        max_len1 = self.cut_len
        pre_1 = get_padding(pre_1, max_len=max_len1)
        post_1 = get_padding(post_1, max_len=max_len1)
        mask_matrix1 = get_mask_matrix(seq_lens, max_len1)
        return index_1, file_1, pre_1, post_1, seq_lens, mask_matrix1, max_len1

    def next_batch(self, batch_size, seed=123456):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self._shuffle(seed=seed + self._epochs_completed)
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        batch_pairs = self.clone_pair[start:end]
        pre_1 = []
        post_1 = []
        pre_2 = []
        post_2 = []
        labels = []
        indices_b_1 = []
        indices_e_1 = []
        indices_b_2 = []
        indices_e_2 = []
        bfs_ind_1 = []
        bfs_ind_2 = []
        dfs_ind_1 = []
        dfs_ind_2 = []
        pre_1_d = []
        pre_2_d = []
        post_1_d = []
        post_2_d = []
        seq_lens = [[], []]
        # print(batch_pairs)
        for pair_data in batch_pairs:
            index1 = pair_data["index1"]
            index2 = pair_data["index2"]
            file1 = pair_data["file1"]
            file2 = pair_data["file2"]
            label = pair_data["label"]
            labels.append(label)

            code_data1 = self.processed_data[index1][file1]
            code_data2 = self.processed_data[index2][file2]

            pre_1.append(code_data1["preorder"])
            post_1.append(code_data1["postorder"])
            indices_b_1.append(code_data1["indices"][0])
            indices_e_1.append(code_data1["indices"][1])
            bfs_ind_1.append(code_data1["bfs_ind"])
            dfs_ind_1.append(code_data1["dfs_ind"])
            len1 = code_data1["len"]
            pre_1_d.append(code_data1["preorder_d"])
            post_1_d.append(code_data1["postorder_d"])

            pre_2.append(code_data2["preorder"])
            post_2.append(code_data2["postorder"])
            indices_b_2.append(code_data2["indices"][0])
            indices_e_2.append(code_data2["indices"][1])
            bfs_ind_2.append(code_data2["bfs_ind"])
            dfs_ind_2.append(code_data2["dfs_ind"])
            len2 = code_data2["len"]
            pre_2_d.append(code_data2["preorder_d"])
            post_2_d.append(code_data2["postorder_d"])

            seq_lens[0].append(len1)
            seq_lens[1].append(len2)

        # pad and build mask
        # max_len1 = min(max(seq_lens[0]), self.cut_len)
        # max_len2 = min(max(seq_lens[1]), self.cut_len)
        max_len1 = self.cut_len
        max_len2 = self.cut_len

        pre_1 = get_padding(pre_1, max_len=max_len1)
        post_1 = get_padding(post_1, max_len=max_len1)
        pre_2 = get_padding(pre_2, max_len=max_len2)
        post_2 = get_padding(post_2, max_len=max_len2)

        mask_matrix1 = get_mask_matrix(seq_lens[0], max_len1)
        mask_matrix2 = get_mask_matrix(seq_lens[1], max_len2)

        return [max_len1, max_len2], [pre_1, post_1, pre_2, post_2], seq_lens, [mask_matrix1, mask_matrix2], labels, \
               [[indices_b_1, indices_e_1, indices_b_2, indices_e_2], [bfs_ind_1, bfs_ind_2, dfs_ind_1, dfs_ind_2],
                [pre_1_d, post_1_d, pre_2_d, post_2_d]]