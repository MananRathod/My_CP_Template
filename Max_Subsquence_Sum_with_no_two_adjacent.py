class SegmentTree:
    def __init__(self, n):
        self.nums = [0]*(4*n+5)
        self.a1 = [0]*(4*n+5) # include l and r
        self.a2 = [0]*(4*n+5) # include l
        self.a3 = [0]*(4*n+5) # include r
        self.a4 = [0]*(4*n+5) # not include l and r
        self.right = n

    def update(self, index, num):
        self._update(0, 0, self.right, index, num)
    def _update(self, index, curr_left, curr_right, target_index, num):
        if curr_left == curr_right:
            self.nums[index] = num
            self.a1[index] = num
            self.a2[index] = 0
            self.a3[index] = 0
            self.a4[index] = 0
            return

        mid = (curr_left + curr_right) >> 1
        if target_index > mid:
            self._update(index*2+2, mid+1, curr_right, target_index, num)
        else:
            self._update(index*2+1, curr_left, mid, target_index, num)

        self.a1[index] = max(
            self.a1[index*2+1] + self.a3[index*2+2],
            self.a2[index*2+1] + self.a1[index*2+2],
            self.a2[index*2+1] + self.a3[index*2+2]
        )
        self.a2[index] = max(
            self.a1[index*2+1] + self.a4[index*2+2],
            self.a2[index*2+1] + self.a2[index*2+2],
            self.a2[index*2+1] + self.a4[index*2+2]
        )
        self.a3[index] = max(
            self.a3[index*2+1] + self.a3[index*2+2],
            self.a4[index*2+1] + self.a1[index*2+2],
            self.a4[index*2+1] + self.a3[index*2+2]
        )
        self.a4[index] = max(
            self.a3[index*2+1] + self.a4[index*2+2],
            self.a4[index*2+1] + self.a2[index*2+2],
            self.a4[index*2+1] + self.a4[index*2+2]
        )
        return
    def query(self):
        return max(self.a1[0], self.a2[0], self.a3[0], self.a4[0])
