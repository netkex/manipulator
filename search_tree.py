from heapq import heappop, heappush, heapify

class DefaultSearchTree:  

    def __init__(self):
        self._open = []  
        self._closed = set()  

    def __len__(self):
        return len(self._open) + len(self._closed)

    def open_is_empty(self):
        return len(self._open) == 0

    def add_to_open(self, item):
        heappush(self._open, item)

    def get_best_node_from_open(self):
        while len(self._open) > 0:
            item = heappop(self._open)
            if not self.was_expanded(item):
                return item
        return None


    def add_to_closed(self, item):
        self._closed.add(item)

    def was_expanded(self, item):
        return item in self._closed

    @property
    def OPEN(self):
        return self._open

    @property
    def CLOSED(self):
        return self._closed
