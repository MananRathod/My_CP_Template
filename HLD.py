class SegmentTree:
    def __init__(self, array, func=max):
        self.n = len(array)
        self.size = 2**(int(log2(self.n-1))+1) if self.n != 1 else 1
        self.func = func
        self.default = 0 if self.func != min else inf
        self.data = [self.default] * (2 * self.size)
        self.process(array)
    def process(self, array):
        self.data[self.size : self.size+self.n] = array
        for i in range(self.size-1, -1, -1):
            self.data[i] = self.func(self.data[2*i], self.data[2*i+1])
    def query(self, alpha, omega):
        if alpha == omega:
            return self.data[alpha + self.size]
        res = self.default
        alpha += self.size
        omega += self.size + 1
        while alpha < omega:
            if alpha & 1:
                res = self.func(res, self.data[alpha])
                alpha += 1
            if omega & 1:
                omega -= 1
                res = self.func(res, self.data[omega])
            alpha >>= 1
            omega >>= 1
        return res
    def update(self, index, value):
        index += self.size
        self.data[index] = value
        index >>= 1
        while index:
            self.data[index] = self.func(self.data[2*index], self.data[2*index+1])
            index >>= 1

class RangeQuery:
    def __init__(self, data, func=min):
        self.func = func
        self._data = _data = [list(data)]
        i, n = 1, len(_data[0])
        while 2 * i <= n:
            prev = _data[-1]
            _data.append([func(prev[j], prev[j + i]) for j in range(n - 2 * i + 1)])
            i <<= 1

    def query(self, begin, end):
        depth = (end - begin).bit_length() - 1
        return self.func(self._data[depth][begin], self._data[depth][end - (1 << depth)])


class LCA:
    def __init__(self, root, graph):
        self.time = [-1] * len(graph)
        self.path = [-1] * len(graph)
        P = [-1] * len(graph)
        t = -1
        dfs = [root]
        while dfs:
            node = dfs.pop()
            self.path[t] = P[node]
            self.time[node] = t = t + 1
            for nei in graph[node]:
                if self.time[nei] == -1:
                    P[nei] = node
                    dfs.append(nei)
        self.rmq = RangeQuery(self.time[node] for node in self.path)

    def __call__(self, a, b):
        if a == b:
            return a
        a = self.time[a]
        b = self.time[b]
        if a > b:
            a, b = b, a
        return self.path[self.rmq.query(a, b)]

class HLD:
    def __init__(self, adj, data, func = max, root = 0):
        self.n = len(adj)
        self.head = [i for i in range(self.n)]
        self.compressed_val = [0] * self.n
        self.times = [0] * self.n
        self.func = func
        self.root = root
        self.subtree_size = [1] * self.n
        self.adj = adj
        self.data = data
        self.level = [0] * self.n
        self.par = [-1] * self.n
        self.s = []
        self.lca = LCA(self.root, self.adj)
        self.calculate_subtree_size()
        self.set_values()
        self.build_segment_tree()
    def calculate_subtree_size(self):
        # use of bfs to calc subtree_size
        q = deque()
        q.append(self.root)
        vis = [False] * self.n
        vis[self.root] = True
        leaves = []
        while q:
            k = len(q)
            for i in range(k):
                node = q.popleft()
                f = 1
                for nbr in self.adj[node]:
                    if(not vis[nbr]):
                        f = 0
                        vis[nbr] = True
                        q.append(nbr)
                        self.level[nbr] = self.level[node] + 1
                        self.par[nbr] = node
                if(f):
                    leaves.append(node)
        # now go your way up
        for node in leaves:
            q.append(node)
            vis[node] = False
        t = [0] * self.n
        while q:
            k = len(q)
            for i in range(k):
                node = q.popleft()
                for nbr in self.adj[node]:
                    if(vis[nbr]):
                        t[nbr] += 1
                        self.subtree_size[nbr] += self.subtree_size[node]
                        if(nbr != self.root and t[nbr] == len(self.adj[nbr]) - 1):
                            q.append(nbr)
                            vis[nbr] = False
    def set_values(self):
        # use of bfs to set values
        # as iterative is better than recursive
        current_compressed_value = 0
        q = deque()
        q.append([self.root, 1]) # [node, head_or_not]
        vis = [False] * self.n
        vis[self.root] = True
        compressed_val_order = []
        compressed_val_max_depth = [0] * self.n
        while q:
            node, flag = q.popleft()
            if(flag): # means he is the head of it's segment
                self.compressed_val[node] = current_compressed_value
                compressed_val_order.append(node)
                current_compressed_value += 1
            tmp = []
            for nbr in self.adj[node]:
                if(not vis[nbr]):
                    tmp.append([self.subtree_size[nbr], nbr])
            tmp.sort(reverse = True)
            for i in range(len(tmp)):
                ignore_subtree_size, nbr = tmp[i]
                vis[nbr] = True
                if(i == 0):
                    q.append([nbr, 0])
                    self.head[nbr] = self.head[node]
                    self.times[nbr] = self.times[node] + 1
                    compressed_val_max_depth[self.head[node]] = self.times[nbr]
                else:
                    q.append([nbr, 1])
        cur_compreesed_val = 0
        for i in range(len(compressed_val_order)):
            node = compressed_val_order[i]
            self.compressed_val[node] = cur_compreesed_val
            cur_compreesed_val += compressed_val_max_depth[node] + 1
    def build_segment_tree(self):
        helper = [0] * self.n
        for i in range(self.n):
            pos = self.compressed_val[self.head[i]] + self.times[i]
            helper[pos] = self.data[i]
        self.s = SegmentTree(helper, self.func)
    def update(self, index, value):
        pos = self.compressed_val[self.head[index]] + self.times[index]
        self.data[index] = value
        self.s.update(pos, value)
    def query(self, a, b):
        ans = 0 if self.func != min else inf
        while 1:
            # first check if they are on the same segment or not
            seg1 = self.head[a]
            seg2 = self.head[b]
            if(seg1 == seg2):
                pos1 = self.compressed_val[self.head[a]] + self.times[a]
                pos2 = self.compressed_val[self.head[b]] + self.times[b]
                left = min(pos1, pos2)
                right = pos1 + pos2 - left
                ans = self.func(ans, self.s.query(left, right))
                break
            # now check which of them is on more depth and then bring it up
            if(self.level[a] < self.level[b]):
                a, b = b, a
            # now move the node a up
            head = self.head[a]
            l = self.lca.__call__(a, b)
            h = head
            if(self.level[l] > self.level[head]):
                h = l
            if(a != head):
                pos1 = self.compressed_val[self.head[a]] + self.times[a]
                pos2 = self.compressed_val[self.head[h]] + self.times[h]
                left = min(pos1, pos2)
                right = pos1 + pos2 - left
                ans = self.func(ans, self.s.query(left, right))
                a = h
            else:
            # now move a one place up
                parent_of_a = self.par[a]
                ans = self.func(ans, self.data[parent_of_a])
                ans = self.func(ans, self.data[a])
                a = parent_of_a
        return ans
