#**** MY CP Template ****#
#import sys, os, io
#input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline
from queue import PriorityQueue # len(pq) == pq.qsize() here
from collections import Counter
from collections import defaultdict
from datetime import date,time
import heapq
from heapq import heapify,heappush,heappop,nlargest,nsmallest,heappushpop,heapreplace
from math import gcd,floor,sqrt,log,ceil,inf
import math
from collections import *
from random import randint
from collections import deque
from itertools import permutations
from math import log2
from bisect import bisect_left
from bisect import bisect_right
#**** Macros ****#
# For fast IO
import os
import sys
from io import BytesIO, IOBase
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def _init_(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def _init_(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")
def li(): return list(map(int, input().split()))
def mp(): return map(int, input().split())
def inp(): return int(input())
def st(): return list(input().strip())
def out(arr): return sys.stdout.write(" ".join(map(str, arr)) + "\n")
def pr(n): return sys.stdout.write(str(n)+"\n")
def prl(n): return sys.stdout.write(str(n)+" ")
M = 1000000007
mod = 998244353
from os import path
if(path.exists('Input.txt')):
    sys.stdin = open("Input.txt","r")
    sys.stdout = open("Output.txt","w")
#**** Macros ****#
if 'PyPy' in sys.version:
    from _continuation import continulet
else:
    import threading
def main():
    pass
if _name_ == '_main_':
    if 'PyPy' in sys.version:
        def bootstrap(cont):
            call, arg = cont.switch()
            while True:
                call, arg = cont.switch(to=continulet(lambda _, f, args: f(*args), call, arg))
        cont = continulet(bootstrap)
        cont.switch()
        main()
    else:
        sys.setrecursionlimit(1 << 30)
        threading.stack_size(1 << 27)
        main_thread = threading.Thread(target=main)
        main_thread.start()
        main_thread.join()
INF = float('inf') 
yes, no = "YES", "NO"
#**** Adv DS ****#
class DSU:
    def _init_(self, n):
        self.parent = [*range(n+1)]
        self.size = [1]*(n+1)
        self.min, self.max = [*range(n+1)], [*range(n+1)]
        self.count = n
    def find(self, a):
        if self.parent[a] == a:
            return a
        x = a
        while a != self.parent[a]:
            a = self.parent[a]
        while x != self.parent[x]:
            self.parent[x], x = a, self.parent[x]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] > self.size[b]:
                a, b = b, a
            # self.parent[a] = max(self.parent[a],self.parent[b])
            # self.parent[b] = max(self.parent[a],self.parent[b])
            self.parent[a] = b
            self.size[b] += self.size[a]
            self.min[b] = min(self.min[a], self.min[b])
            self.max[b] = max(self.max[a], self.max[b])
            self.count -= 1
    def countSets(self):
        return self.count
class TrieNode:
    def _init_(self): #constructor
        self.children = [None] * 26 #creates the 26 child with inital None value
        self.isEndOfWord = False #This helps in marking the end of the words i.e.[Present or not]
class Trie: #Remember: You can update the insert and search function according to the question
    def _init_(self):
        self.root = self.getNode()
    def getNode(self):
        return TrieNode()
    def _charToIndex(self, ch): #converts the char to an index
        return ord(ch) - ord('a')
    def insert(self, key): #inserts a key/string into the tree/trie
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]
        pCrawl.isEndOfWord = True
    def search(self, key): #searches for the specific key/string in the tree/trie
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]
        return pCrawl != None and pCrawl.isEndOfWord
    def delete(self, key,i=0):
    	pCrawl = self.root
    	if(pCrawl == None):
    		return pCrawl
    	length = len(key)
    	if(i == length):
    		if(pCrawl.isEndOfWord):
    			pCrawl.isEndOfWord=False
    		if(self.isEmpty(pCrawl)):
    			pCrawl=None
    		return pCrawl
    	index=self._charToIndex(key[i])
    	pCrawl.children[index]=self.delete(key,i+1)
    	if(self.isEmpty(pCrawl) and pCrawl.isEndOfWord==False):
    		pCrawl=None
    	return pCrawl
    def isEmpty(self,root1):
    	for i in root1.children:
    		if(i != None):
    			return False
    	return True
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
class fenwTree: #used for prefix operations
	def _init_(self,a,func=lambda a,b:a+b):
		self.a=a
		self.func=func
		self.arr=[0]*(len(a)+1)
		for i in range(len(a)):
			self.buildTree(i,a[i])
	def buildTree(self,idx,val):
		idx+=1 #make it a 1-based indexing
		while idx<=len(self.a):
			self.arr[idx]=self.func(self.arr[idx],val)
			idx+=idx&(-idx)
	def update(self,idx,val):
		idx+=1 #make it a 1-based indexing
		diff=val-self.a[idx-1]
		self.a[idx-1]=val
		while idx<=len(self.a):
			self.arr[idx]=self.func(self.arr[idx],diff)
			idx+=idx&(-idx)
	def query(self,idx):
		idx+=1
		ans=0
		while idx>0:
			ans=self.func(ans,self.arr[idx])
			idx-=idx&(-idx)
		return ans
def addUndirectedEdge(adj,u,v):
	adj[u].append(v)
	adj[v].append(u)
def addDirectedEdge(adj,u,v):
	adj[u].append(v)
def dijkstra(adj,src,n):
	parent=[-1 for i in range(n)]
	dist=[inf for i in range(n)]
	dist[src]=0
	heap=[]
	heapify(heap)
	heappush(heap,[0,src])
	while heap:
		curDist,v=heappop(heap)
		for nbr,length in adj[v]:
			if(curDist+length<dist[nbr]):
				dist[nbr]=curDist+length
				parent[nbr]=v
				heappush(heap,[dist[nbr],nbr])
	return dist,parent
class Sparse: #used when array is fixed and overlapping allowed
    def _init_(self, a, func=min):
        self.a = a
        self.n = len(a)
        self.k = int(log2(self.n)) + 1
        self.func = func
        self.lookup = [[0] * self.k for _ in range(self.n)]
        self.process()
    def process(self):
        for i in range(self.n):
            self.lookup[i][0] = self.a[i]
        j = 1
        while (1 << j) <= self.n:
            i = 0
            while (i + (1 << j) - 1) < self.n:
                self.lookup[i][j] = self.func(self.lookup[i][j - 1], self.lookup[i + (1 << (j - 1))][j - 1])
                i += 1
            j += 1
    def query(self, alpha, omega):
        x = int(log2(omega - alpha + 1))
        return self.func(self.lookup[alpha][x], self.lookup[omega - (1 << x) + 1][x])
class mydict:
    def _init_(self, func=lambda: 0):
        self.random = randint(0, 1 << 32)
        self.default = func
        self.dict = {}
    def _getitem_(self, key):
        mykey = self.random ^ key
        if mykey not in self.dict:
            self.dict[mykey] = self.default()
        return self.dict[mykey]
    def get(self, key, default):
        mykey = self.random ^ key
        if mykey not in self.dict:
            return default
        return self.dict[mykey]
    def _setitem_(self, key, item):
        mykey = self.random ^ key
        self.dict[mykey] = item
    def getkeys(self):
        return [self.random ^ i for i in self.dict]
    def _str_(self):
        return f'{[(self.random ^ i, self.dict[i]) for i in self.dict]}'
#**** Adv DS ****#
import operator as op
from functools import reduce
from math import *
def NcR(n, r):
    p = 1
    k = 1
    if (n - r < r):
        r = n - r
    if (r != 0):
        while (r):
            p *= n
            k *= r
            m = gcd(p, k)
            p //= m
            k //= m
            n -= 1
            r -= 1
            p%=M
    else:
        p = 1
    return p
def fact(n):
	return math.factorial(n)
def gcd(a,b):
    if (a==0):
        return b
    return gcd(b % a, a)
def lcm(a, b):
    return (a * b)// gcd(a, b)
def decimalToBinary(n):
    return bin(n).replace("0b", "")
def binaryToDecimal(n):
    return int(n,2)
def SieveOfEratosthenes(n):
    isPrime = [True]*(n+1)
    isPrime[0] = isPrime[1] = False
    primes = []
    for i in range(2, n+1):
        if not isPrime[i]:continue
        primes += [i]
        for j in range(i*i, n+1, i):
            isPrime[j] = False
    return primes
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        if array[mid] == target:
            return mid
        elif array[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    return -1
def gcd_of_array(arr):
	ans=arr[0]
	for i in range(1,len(arr)):
		ans=gcd(ans,arr[i])
	return ans
def isPrime(n):
	if(n==2):
		return True
	if (n % 2 == 0 and n != 2) or n < 2:
	    return False
	i = 3
	while i * i <= n:
	    if n % i == 0:
	        return False
	    i += 2
	return True
def isPowerOfTwo(n):
    return n&(n-1)==0
def firstSetBit(n):
	return n&~(n-1)
def expo(a, n, k): 
    ans = 1
    while(n > 0):
        last_bit = n&1
        if(last_bit):
            ans = (ans*a)%k
        a = (a*a)%k
        n = n >> 1
    return ans
def prefix_sum(arr):
	n=len(arr)
	l = [arr[0]]
	for i in range(1,n):
		x = l[-1]
		l.append(x+arr[i])
	return l
def sqrtSearch(low, high, N) : #use this function to find the square root instead of the inbuilt one
    while (low <= high) :
        mid = (low + high) // 2;
        if ((mid * mid <= N) and ((mid + 1) * (mid + 1) > N)) :
            return mid;
        elif (mid * mid < N) :
            low = mid+1
        else :
            high = mid-1
    return high;
#**** MY CP Template ****#	

#**** KEEP IN MIND ****#
'''
  WRITE STUFF DOWN
  READ THE QUESTION PROPERLY [Many times if required]
  FIRST SOLVE ON PAPER AND GET THE CORRECT APPROACH THEN ONLY START CODING
  [This will save alot of time as well as lower incorrect submissions]
  DON'T GET STUCK ON ONE APPROACH [Try to think differently]
  TRY TO SOLVE SUBPROBLEMS OR FOR N==2,3,4.. FIRST THEN YOU MIGHT START SEEING SOMETHING
  DON'T REPEAT THE SAME MISTAKES DONE IN PREVIOUS CONTESTS
  1. Read the problem carefully
  2. Understand what is being asked to find out
  3. Think how could that be found
  4. Recall if you could use some algos to find out the needed
  5. rank & ratings <<<<< Problem solving 
  6. Goal should always be solving more number of problems rather than rank,ratings,speed etc
'''
#**** KEEP IN MIND ****#	  	        
		
# Solver function
def solve():
	
			 
# Main 
for _ in range(inp()):
	#prl(f"Case #{_+1}:")
	solve()
	
#**** NOTE ****#	
'''Using Stack and while loop may save the execution time of the program
 If the array contains large numbers then don't do this --> a=list(set(a))
 To reduce execution time - Iterative sol > Recursive sol
 Graph bfs solutions are faster than dfs solutions
 Sparese Table and Fenwick tree are faster than segTree
 List comprehension is faster than for loops
 if solution relies heavily on data structure then submit the sol on python 3
'''
#**** NOTE ****#
