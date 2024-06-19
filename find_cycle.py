def find_cycle(adj, n, src):
    stack = []
    stack2 = []
    vis = [False] * n
    stack.append(src)
    stack2.append(src)
    vis[src] = True
    par = [-1] * n
    start = -1
    end = -1
    while stack:
        node = stack.pop()
        vis[node] = True
        orig_vis[node] = True
        flag = False
        for nbr in adj[node]:
            if(nbr == par[node]):
                continue
            if(vis[nbr]):
                start = node
                end = nbr
                break
            stack.append(nbr)
            stack2.append(nbr)
            par[nbr] = node
            flag = True
        if(start != -1):
            break
        if(flag == False):
            while stack and stack2 and stack[-1] != stack2[-1]:
                vis[stack2.pop()] = False
    if(start == -1):
        return False, []
    path = []
    path.append(start)
    start = par[start]
    while 1:
        path.append(start)
        if(start == end):
            break
        start = par[start]
    path.append(path[0])
    return True, path
