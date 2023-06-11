global n
par=mydict()
lvl=mydict()

def bfs(adj,src):
	global n,par,lvl
	vis=[False for i in range(n+1)]
	q=deque()
	q.append(1)
	cnt=0
	vis[1]=True
	while q:
		x=len(q)
		cnt+=1
		for i in range(x):
			y=q.popleft()
			lvl[y]=cnt
			for j in adj[y]:
				if(vis[j]==False):
					q.append(j)
					vis[j]=True
					par[j]=y

def dfsIter(adj,src):
	global n,par,lvl
	global timeIn,timeOut
	stack=[]
	stack2=[]
	counter=0
	stack.append(src)
	stack2.append(src)
	vis=[False for i in range(n+1)]
	vis[src]=True
	while stack:
		x=stack.pop()
		timeIn[x]=counter
		change=0
		counter+=1
		for i in adj[x]:
			if(vis[i]==False):
				stack.append(i)
				stack2.append(i)
				change+=1
				vis[i]=True
		if(change==0):
			if(len(stack)==0):
				while stack2:
					timeOut[stack2.pop()]=counter
			else:
				while 1:
					if(stack2[-1]==stack[-1]):
						break
					timeOut[stack2.pop()]=counter
