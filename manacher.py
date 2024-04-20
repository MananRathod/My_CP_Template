# manacher algo
class manacher:
    def __init__(self, s):
        self.t_string = ['#']
        for char in s:
            self.t_string.append(char)
            self.t_string.append('#')
        self.answer = [0] * len(self.t_string)
        self.build()
    def build(self):
        n = len(self.t_string)
        ll = -1
        rr = 0
        for i in range(n):
            self.answer[i] = max(0, min(rr - i, self.answer[ll + rr - i]))
            cur_ans = self.answer[i]
            l = i - cur_ans
            r = i + cur_ans
            while l >= 0 and r < n:
                if(self.t_string[l] == self.t_string[r]):
                    cur_ans += 1
                    l -= 1
                    r += 1
                else:
                    break
            self.answer[i] = cur_ans
            if(i + self.answer[i] > rr):
                ll = i - self.answer[i]
                rr = i + self.answer[i]
    def isPossibleRange(self, l, r):
        # nidx = idx * 2 + 1
        idx = (l + r) >> 1
        nidx = idx * 2 + 1
        if((r - l + 1) % 2 == 0):
            nidx += 1
        ans = self.answer[nidx]
        lidx = l * 2 + 1
        on_left = nidx - lidx + 1
        return ans >= on_left
