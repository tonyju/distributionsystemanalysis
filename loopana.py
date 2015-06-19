import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
#simple system data
#br.no, Rc.nd, Sn.nd, Br.r, Br.x, Sn.load.p, Sn.load.q 
treedisnet=[[1,0,1,0.0922,0.0470,100.00,60.00],
[2,1,2,0.04930,0.02511,90.00,40.00],
[3,2,3,0.03660,0.01864,120.00,80.00]
]
loopdisnet=[
            [4,1,3,0.03660,0.01864,0.00,0.00]
            ]
loopdisnet=[]
labels={}
allnodelist=[]
for i in range(4):
    allnodelist.append(i)
for i in range(4):
    labels[i]=str(i)
#33 buses system data
#br.no, Rc.nd, Sn.nd, Br.r, Br.x, Sn.load.p, Sn.load.q 
'''treedisnet=[[1,0,1,0.0922,0.0470,100.00,60.00],
[2,1,2,0.4930,0.2511,90.00,40.00],
[3,2,3,0.3660,0.1864,120.00,80.00],
[4,3,4,0.3811,0.1941,60.00,30.00],
[5,4,5,0.8190,0.7070,60.00,20.00],
[6,5,6,0.1872,0.6188,200.00,100.00],
[7,6,7,0.7114,0.2351,200.00,100.00],
[8,7,8,1.0300,0.7400,60.00,20.00],
[9,8,9,1.0440,0.7400,60.00,20.00],
[10,9,10,0.1966,0.0650,45.00,30.00],
[11,10,11,0.3744,0.1238,60.00,35.00],
[12,11,12,1.4680,1.1550,60.00,35.00],
[13,12,13,0.5416,0.7129,120.00,80.00],
[14,13,14,0.5910,0.5260,60.00,10.00],
[15,14,15,0.7463,0.5450,60.00,20.00],
[16,15,16,1.2890,0.7129,120.00,80.00],
[17,16,17,0.7320,0.5740,90.00,40.00],
[18,1,18,0.1640,0.1565,90.00,40.00],
[19,18,19,1.5042,1.3554,90.00,40.00],
[20,19,20,0.4095,0.4784,90.00,40.00],
[21,20,21,0.2089,0.9373,90.00,40.00],
[22,2,22,0.4512,0.3083,90.00,50.00],
[23,22,23,0.8980,0.7091,420.00,200.00],
[24,23,24,0.8960,0.7071,420.00,200.00],
[25,5,25,0.2030,0.1034,60.00,25.00],
[26,25,26,0.2842,0.1447,60.00,25.00],
[27,26,27,1.0590,0.9337,60.00,20.00],
[28,27,28,0.8042,0.7006,120.00,70.00],
[29,28,29,0.5075,0.2585,200.00,600.00],
[30,29,30,0.9744,0.9630,150.00,70.00],
[31,30,31,0.3105,0.3619,210.00,100.00],
[32,31,32,0.3410,0.5302,60.00,40.00]
]
loopdisnet=[
            [33,7,20,2.0000,2.0000,0.00,0.00],
            [34,8,14,2.0000,2.0000,0.00,0.00],
            [35,11,21,2.0000,2.0000,0.00,0.00],
            [36,17,32,0.5000,0.5000,0.00,0.00],
            [37,24,28,0.5000,0.5000,0.00,0.00]
             ]
loopdisnet=[[37,24,28,0.1000,0.1000,0.00,0.00]]
labels={}
allnodelist=[]
for i in range(33):
    allnodelist.append(i)
for i in range(33):
    labels[i]=str(i)'''

#voltage source,-1 denotes the ground nodes
vsource=[[-1,0,12.66e3,0.0]]
n3branch=len(vsource)
branchdict={}
loaddict={}
G=nx.Graph()
treeedgelist=[]
loopedgelist=[]
n1branch=0
n2branch=0
for item in range(len(treedisnet)):
    branchinfo=treedisnet[item]
    nfr=branchinfo[1]
    nto=branchinfo[2]
    treeedgelist.append((nfr,nto))
    G.add_edge(nfr, nto)
    branchdict[(nfr,nto)]=[branchinfo[3],branchinfo[4],n1branch]
    branchdict[(nto,nfr)]=[branchinfo[3],branchinfo[4],n1branch]
    loaddict[(nto,-1)]=[branchinfo[5]*1000.0,branchinfo[6]*1000.0,n2branch]
    n1branch=n1branch+1
    n2branch=n2branch+1
for item in range(len(loopdisnet)):
    branchinfo=loopdisnet[item]
    nfr=branchinfo[1]
    nto=branchinfo[2]
    G.add_edge(nfr, nto)
    loopedgelist.append((nfr,nto))
    branchdict[(nfr,nto)]=[branchinfo[3],branchinfo[4],n1branch]
    branchdict[(nto,nfr)]=[branchinfo[3],branchinfo[4],n1branch]
    n1branch=n1branch+1
rb=np.zeros((n1branch,n1branch))
xb=np.zeros((n1branch,n1branch))
for itembranch in branchdict:
    rb[branchdict[itembranch][2],branchdict[itembranch][2]]=branchdict[itembranch][0]
    xb[branchdict[itembranch][2],branchdict[itembranch][2]]=branchdict[itembranch][1]
il1re0=np.zeros((n2branch,1))
il1im0=np.zeros((n2branch,1))
for itemload in loaddict:
    il1re0[loaddict[itemload][2]]=loaddict[itemload][0]/12.66
    il1im0[loaddict[itemload][2]]=-loaddict[itemload][1]/12.66
#print rb
#print xb
# some math labels
pos=nx.graphviz_layout(G)

nx.draw_networkx_nodes(G,pos,
                       nodelist=allnodelist,
                       node_color='b',
                       node_size=500,
                   alpha=0.8)
nx.draw_networkx_edges(G,pos,
                       edgelist=treeedgelist,
                       width=8,alpha=0.5,edge_color='b')
nx.draw_networkx_edges(G,pos,
                       edgelist=loopedgelist,
                       width=8,alpha=0.5,edge_color='r')
nx.draw_networkx_labels(G,pos,labels,font_size=16)
plt.axis('off')
plt.savefig("labels_and_colors.svg") # save as png
#plt.show()

nloop=0

'''Breadth First Search'''
#print(nx.bfs_successors(G,0))
source=0
visited=set([source])
currentLevel = 0
dlevel=defaultdict(list)
nlevel=0
dlevel[source]=nlevel
nlevel=nlevel+1
stack = [(source,iter(G[source]))]
samelevelloopedges=[]
samelevelloopdict={}
treeedges=[]
treedict={}
nosamelevelloopedges=[]
loopedges=[]

#print stack
while stack:
    parent,children = stack[0]
    try:
        child = next(children)
        #print children
        if child not in visited:
            dlevel[child]=nlevel
            #print parent,child
            visited.add(child)
            treeedges.append((parent,child))
            stack.append((child,iter(G[child])))
        elif dlevel[child]==dlevel[parent]:
            if not samelevelloopdict.has_key((parent,child)):
                samelevelloopedges.append((parent,child))
                samelevelloopdict[(parent,child)]=1
                samelevelloopdict[(child,parent)]=1
                loopedges.append((parent,child))
                nloop=nloop+1
        elif dlevel[child]>dlevel[parent]:
            nosamelevelloopedges.append((parent,child))
            loopedges.append((parent,child))
            nloop=nloop+1
    except StopIteration:
        stack.pop(0)
        nlevel=nlevel+1
#print dlevel
edges=nx.bfs_edges(G,0)
for s,t in nx.bfs_edges(G,0):
    pass#print s,t
prenode=dict((t,s) for s,t in treeedges)#bfs_predecessors
d = defaultdict(list)
for s,t in treeedges:
    d[s].append(t)#bfs_successors
Bl11=np.zeros((n2branch,n1branch))
print n2branch,n1branch
nload=0
source=0
for loaditem in loaddict:
    nload=nload+1
    nfr,nto=loaditem
    print nfr,nto
    pre=prenode[nfr]
    Bl11[loaddict[loaditem][2],branchdict[pre,nfr][2]]=1.0
    if not pre==source:
        preold=pre
        pre=prenode[pre]
        #print pre,preold
        Bl11[loaddict[loaditem][2],branchdict[pre,preold][2]]
print Bl11    
#print nload

#print d#bfs_successors
Bl21=np.zeros((nloop,n1branch))
for i in range(len(loopedges)):
    (nfr,nto)=loopedges[i]
    Bl21[i,branchdict[nfr,nto][2]]=1.0
    pre1=prenode[nfr]
    pre2=prenode[nto]
    print pre1,pre2
    Bl21[i,branchdict[nfr,pre1][2]]=1.0
    Bl21[i,branchdict[nto,pre2][2]]=-1.0
    if not pre1==pre2:
        Bl21[i,branchdict[nfr,pre1][2]]=1.0
        Bl21[i,branchdict[nto,pre2][2]]=-1.0
        nfr=pre1
        nto=pre2
        pre1=prenode[nfr]
        pre2=prenode[nto]              
print Bl21
rb=np.mat(rb)
xb=np.mat(xb)
Bl21=np.mat(Bl21)
Bl11=np.mat(Bl11)
il1re0=np.mat(il1re0)
il1im0=np.mat(il1im0)
print il1re0
print il1im0
  
rht1=-Bl21*rb*Bl11.T*il1re0+Bl21*xb*Bl11.T*il1im0
rht2=-Bl21*xb*Bl11.T*il1re0-Bl21*rb*Bl11.T*il1im0

eleA11=Bl21*rb*Bl21.T
eleA12=-Bl21*xb*Bl21.T
eleA21=Bl21*xb*Bl21.T
eleA22=Bl21*rb*Bl21.T
eleA1=np.hstack((eleA11,eleA12))
eleA2=np.hstack((eleA21,eleA22))
eleA=np.vstack((eleA1,eleA2))
rht=np.vstack((rht1,rht2))
#print eleA
res=np.linalg.solve(eleA, rht)
il2re=res[0:nloop]
il2im=res[nloop:2*nloop]
print il2re,il2im

vb1re=rb*Bl11.T*il1re0+rb*Bl21.T*il2re-xb*Bl11.T*il1im0-xb*Bl21.T*il2im
vb1im=xb*Bl11.T*il1re0+xb*Bl21.T*il2re+rb*Bl11.T*il1im0+rb*Bl21.T*il2im


Bl12=np.mat(np.eye(n2branch,n2branch))
Bl13=np.mat(np.ones((n2branch,n3branch)))
vb3re=-12.66e3
vb3im=0

vb2re=-Bl13*vb3re-Bl11*vb1re
vb2im=-Bl13*vb3im-Bl11*vb1im
#print vb2re
#print vb2im
print np.abs(vb2re+1j*vb2im)/12.66e3

#print(nx.bfs_predecessors(G,0))
#nx.bfs_tree(G,0)
#print loopedges
#print treeedges

for iter in range(20):
    for itemload in loaddict:
        v=vb2re[loaddict[itemload][2]]+1j*vb2im[loaddict[itemload][2]]
        #print np.abs(v)
        il1re0[loaddict[itemload][2]]=np.real(np.conj((loaddict[itemload][0]+1j*loaddict[itemload][1])/(v)))*1000
        il1im0[loaddict[itemload][2]]=np.imag(np.conj((loaddict[itemload][0]+1j*loaddict[itemload][1])/(v)))*1000
        #print iter,itemload,il1re0[loaddict[itemload][2]],il1im0[loaddict[itemload][2]]
    rht1=-Bl21*rb*Bl11.T*il1re0+Bl21*xb*Bl11.T*il1im0
    rht2=-Bl21*xb*Bl11.T*il1re0-Bl21*rb*Bl11.T*il1im0
    
    eleA11=Bl21*rb*Bl21.T
    eleA12=-Bl21*xb*Bl21.T
    eleA21=Bl21*xb*Bl21.T
    eleA22=Bl21*rb*Bl21.T
    eleA1=np.hstack((eleA11,eleA12))
    eleA2=np.hstack((eleA21,eleA22))
    eleA=np.vstack((eleA1,eleA2))
    rht=np.vstack((rht1,rht2))
    #print eleA
    res=np.linalg.inv(eleA)*rht
    #res=np.linalg.solve(eleA, rht)
    il2re=res[0:nloop]
    il2im=res[nloop:2*nloop]
    print iter,il2re,il2im
    vb1re=rb*Bl11.T*il1re0+rb*Bl21.T*il2re-xb*Bl11.T*il1im0-xb*Bl21.T*il2im
    vb1im=xb*Bl11.T*il1re0+xb*Bl21.T*il2re+rb*Bl11.T*il1im0+rb*Bl21.T*il2im
    vb1res=np.abs(vb1re+1j*vb1im)/12.66e3
    print iter,vb1res
    
    vb2re=-Bl13*vb3re-Bl11*vb1re
    vb2im=-Bl13*vb3im-Bl11*vb1im
vres=np.abs(vb2re+1j*vb2im)/12.66e3
#print vb1res
for loaditem in loaddict:
    print loaditem,vres[loaddict[loaditem][2]]
#
#
#[[ 0.97633684]
# [ 0.95175011]
# [ 0.95139575]] 7
#