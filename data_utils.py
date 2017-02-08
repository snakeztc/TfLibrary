import numpy as np
import math

# Data feed
class DataLoader(object):
    batch_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    indexes = None
    data_size = None
    name = None
    equal_len_batch = None

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_index):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True):
        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        print("Number of left over sample %d" % (self.data_size-batch_size*self.num_batch))

        # if shuffle and we don't want to group lines, shuffle index
        if shuffle and not self.equal_len_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and self.equal_len_batch:
            self._shuffle_batch_indexes()

        print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None



# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.backward_size = backward_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len-self.backward_size) // self.step_size
            if num_seg > 0:
                cut_start = range(0, num_seg*self.step_size, step_size)
                cut_end = range(self.backward_size, num_seg*self.step_size+self.backward_size, step_size)
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.backward_size-2) +cut_start # since we give up on the seq training idea
                cut_end = range(2, self.backward_size) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = range(2, max_len)

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
               np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None

