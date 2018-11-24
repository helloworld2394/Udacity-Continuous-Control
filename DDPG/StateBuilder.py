# State stack builder
# Copyright 2018 hojun Yoon. All rights reserved.

import numpy as np


class StateBuilder:
    def __init__(self, state_size, frames, workers):
        self._state_size = state_size
        self._frames = frames
        self._workers = workers

        self.state = np.empty([workers, state_size*frames])

    def _stack(self, s):
        tmp = np.empty([self._frames, self._state_size])
        for i in range(self._frames):
            tmp[i] = s
        return tmp.flatten()

    def reset_worker(self, worker_id, s):
        self.state[worker_id] = self._stack(s)

    def reset_state(self, s):
        for i in range(self._workers):
            self.state[i] = self._stack(s[i])

    def append_worker(self, worker_id, s):
        self.state[worker_id] = np.append(s, self.state[worker_id][0:-self._state_size])

    def append(self, s):
        for i in range(self._workers):
            self.append_worker(i, s[i])
