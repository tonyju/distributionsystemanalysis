import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
#69 buses system data
disnet=[
[1,'0','1'],
[2,'1','2'],
[3,'2','2e'],
[4,'2e','3'],
[5,'3','4'],
[6,'4','5'],
[7,'5','6'],
[8,'6','7'],
[9,'7','8'],
[10,'8','9'],
[11,'9','10'],
[12,'10','11'],
[13,'11','12'],
[14,'12','13'],
[15,'13','14'],
[16,'14','15'],
[17,'15','16'],
[18,'16','17'],
[19,'17','18'],
[20,'18','19'],
[21,'19','20'],
[22,'20','21'],
[23,'21','22'],
[24,'22','23'],
[25,'23','24'],
[26,'24','25'],
[27,'25','26'],
[28,'2','27'],
[29,'27','28'],
[30,'28','29'],
[31,'29','30'],
[32,'30','31'],
[33,'31','32'],
[34,'32','33'],
[35,'33','34'],
[36,'2e','27e'],
[37,'27e','28e'],
[38,'28e','65'],
[39,'65','66'],
[40,'66','67'],
[41,'67','68'],
[42,'68','69'],
[43,'69','70'],
[44,'70','88'],
[45,'88','89'],
[46,'89','90'],
[47,'3','35'],
[48,'35','36'],
[49,'36','37'],
[50,'37','38'],
[51,'7','40'],
[52,'40','41'],
[53,'8','42'],
[54,'42','43'],
[55,'43','44'],
[56,'44','45'],
[57,'45','46'],
[58,'46','47'],
[59,'47','48'],
[60,'48','49'],
[61,'49','50'],
[62,'50','51'],
[63,'51','52'],
[64,'52','53'],
[65,'53','54'],
[66,'10','55'],
[67,'55','56'],
[68,'11','57'],
[69,'57','58']
]
nodearray={}
nnode=0
G=nx.Graph()
edgelist=[]
for i in range(len(disnet)):
    fr=disnet[i][1]
    if not nodearray.has_key(fr):
        nodearray[fr]=nnode
        nnode=nnode+1
        
    to=disnet[i][2]
    if not nodearray.has_key(to):
        nodearray[to]=nnode
        nnode=nnode+1
    G.add_edge(nodearray[fr], nodearray[to])
    edgelist.append((nodearray[fr],nodearray[to]))
pos=nx.graphviz_layout(G)
labels={}
allnodelist=[]
for i in range(70):
    allnodelist.append(i)
for i in range(70):
    labels[i]=str(i)
nx.draw_networkx_nodes(G,pos,
                       nodelist=allnodelist,
                       node_color='b',
                       node_size=100,
                   alpha=0.8)
nx.draw_networkx_edges(G,pos,
                       edgelist=edgelist,
                       width=8,alpha=0.5,edge_color='b')
nx.draw_networkx_labels(G,pos,labels,font_size=10)
plt.axis('off')
plt.savefig("labels_and_colors_69.svg") # save as png
treedisnet=[
[1,nodearray['0'],nodearray['1'],0.0005,0.0012,0.00,0.00],
[2,nodearray['1'],nodearray['2'],0.0005,0.0012,0.00,0.00],
[3,nodearray['2'],nodearray['2e'],0.0000,0.0000,0.00,0.00],
[4,nodearray['2e'],nodearray['3'],0.0015,0.0036,0.00,0.00],
[5,nodearray['3'],nodearray['4'],0.0251,0.0294,0.00,0.00],
[6,nodearray['4'],nodearray['5'],0.3660,0,1864,2.60,2.20],
[7,nodearray['5'],nodearray['6'],0.3811,0.1941,40.40,30.00],
[8,nodearray['6'],nodearray['7'],0.0922,0.0470,75.00,54.00],
[9,nodearray['7'],nodearray['8'],0.0493,0.0251,30.00,22.00],
[10,nodearray['8'],nodearray['9'],0.8190,0.2707,28.00,19.00],
[11,nodearray['9'],nodearray['10'],0.1872,0.0619,145.00,104.00],
[12,nodearray['10'],nodearray['11'],0.7114,0.2351,145.00,104.00],
[13,nodearray['11'],nodearray['12'],1.0300,0.3400,8.00,5.00],
[14,nodearray['12'],nodearray['13'],1.0440,0.3450,8.00,5.00],
[15,nodearray['13'],nodearray['14'],1.0580,0.3496,0.00,0.00],
[16,nodearray['14'],nodearray['15'],0.1966,0.0650,45.50,30.00],
[17,nodearray['15'],nodearray['16'],0.3744,0.1238,60.00,35.00],
[18,nodearray['16'],nodearray['17'],0.0047,0.0016,60.00,35.00],
[19,nodearray['17'],nodearray['18'],0.3276,0.1083,0.00,0.00],
[20,nodearray['18'],nodearray['19'],0.2106,0.0696,1.00,0.60],
[21,nodearray['19'],nodearray['20'],0.3416,0.1129,114.00,81.00],
[22,nodearray['20'],nodearray['21'],0.0140,0.0046,5.30,3.50],
[23,nodearray['21'],nodearray['22'],0.1591,0.0526,0.00,0.00],
[24,nodearray['22'],nodearray['23'],0.3463,0.1145,28.00,20.00],
[25,nodearray['23'],nodearray['24'],0.7488,0.2475,0.00,0.00],
[26,nodearray['24'],nodearray['25'],0.3089,0.1021,14.00,10.00],
[27,nodearray['25'],nodearray['26'],0.1732,0.0572,14.00,10.00],
[28,nodearray['2'],nodearray['27'],0.0044,0.0108,26.00,18.60],
[29,nodearray['27'],nodearray['28'],0.0640,0.1565,26.00,18.60],
[30,nodearray['28'],nodearray['29'],0.3978,0.1315,0.00,0.00],
[31,nodearray['29'],nodearray['30'],0.0702,0.0232,0.00,0.00],
[32,nodearray['30'],nodearray['31'],0.3510,0.1160,0.00,0.00],
[33,nodearray['31'],nodearray['32'],0.8390,0.2816,14.00,10.00],
[34,nodearray['32'],nodearray['33'],1.7080,0.5646,19.50,14.00],
[35,nodearray['33'],nodearray['34'],1.4740,0.4873,6.00,4.00],
[36,nodearray['2e'],nodearray['27e'],0.0044,0.0108,26.00,18.55],
[37,nodearray['27e'],nodearray['28e'],0.0640,0.1565,26.00,18.55],
[38,nodearray['28e'],nodearray['65'],0.1053,0.1230,0.00,0.00],
[39,nodearray['65'],nodearray['66'],0.0304,0.0355,24.00,17.00],
[40,nodearray['66'],nodearray['67'],0.0018,0.0021,24.00,17.00],
[41,nodearray['67'],nodearray['68'],0.7283,0.8509,1.20,1.00],
[42,nodearray['68'],nodearray['69'],0.3100,0.3623,0.00,0.00],
[43,nodearray['69'],nodearray['70'],0.0410,0.0478,6.00,4.30],
[44,nodearray['70'],nodearray['88'],0.0092,0.0116,0.00,0.00],
[45,nodearray['88'],nodearray['89'],0.1089,0.1373,39.22,26.30],
[46,nodearray['89'],nodearray['90'],0.0009,0.0012,39.22,26.30],
[47,nodearray['3'],nodearray['35'],0.0034,0.0084,0.00,0.00],
[48,nodearray['35'],nodearray['36'],0.0851,0.2083,79.00,56.40],
[49,nodearray['36'],nodearray['37'],0.2898,0.7091,384.70,274.50],
[50,nodearray['37'],nodearray['38'],0.0822,0.2011,384.70,274.50],
[51,nodearray['7'],nodearray['40'],0.0928,0.0473,40.50,28.30],
[52,nodearray['40'],nodearray['41'],0.3319,0.1114,3.60,2.70],
[53,nodearray['8'],nodearray['42'],0.1740,0.0886,4.35,3.50],
[54,nodearray['42'],nodearray['43'],0.2030,0.1034,26.40,19.00],
[55,nodearray['43'],nodearray['44'],0.2842,0.1447,24.00,17.20],
[56,nodearray['44'],nodearray['45'],0.2813,0.1433,0.00,0.00],
[57,nodearray['45'],nodearray['46'],1.5900,0.5337,0.00,0.00],
[58,nodearray['46'],nodearray['47'],0.7837,0.2630,0.00,0.00],
[59,nodearray['47'],nodearray['48'],0.3042,0.1006,100.00,72.00],
[60,nodearray['48'],nodearray['49'],0.3861,0.1172,0.00,0.00],
[61,nodearray['49'],nodearray['50'],0.5075,0.2585,1244.00,888.00],
[62,nodearray['50'],nodearray['51'],0.0974,0.0496,32.00,23.00],
[63,nodearray['51'],nodearray['52'],0.1450,0.0738,0.00,0.00],
[64,nodearray['52'],nodearray['53'],0.7105,0.3619,227.00,162.00],
[65,nodearray['53'],nodearray['54'],1.0410,0.5302,59.00,42.00],
[66,nodearray['10'],nodearray['55'],0.2012,0.0611,18.00,13.00],
[67,nodearray['55'],nodearray['56'],0.0047,0.0014,18.00,13.00],
[68,nodearray['11'],nodearray['57'],0.7394,0,2444,28.00,22.00],
[69,nodearray['57'],nodearray['58'],0.0047,0.0016,28.00,20.00]
]
loopdisnet=[]

#simple system data
#br.no, Rc.nd, Sn.nd, Br.r, Br.x, Sn.load.p, Sn.load.q 
'''treedisnet=[[1,0,1,0.0922,0.0470,100.00,60.00],
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
T=np.mat(np.zeros((3,3)))
T[0,0]=1.0
T[1,0]=1.0
T[1,1]=1.0
T[2,0]=1.0
T[2,1]=1.0
T[2,2]=1.0'''

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
loopdisnet=[]
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
    il1re0[loaddict[itemload][2]]=loaddict[itemload][0]/12.66e3
    il1im0[loaddict[itemload][2]]=-loaddict[itemload][1]/12.66e3
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
    while not pre==source:
        preold=pre
        pre=prenode[pre]
        print pre,preold
        Bl11[loaddict[loaditem][2],branchdict[pre,preold][2]]=1.0
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
    while not pre1==pre2:
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
#test 3nodes system
zb=rb+1j*xb
#vb=zb*T.T*(il1re0+1j*il1im0)
#print vb,vb1re+1j*vb1im

Bl12=np.mat(np.eye(n2branch,n2branch))
Bl13=np.mat(np.ones((n2branch,n3branch)))
vb3re=-12.66e3
vb3im=0

vb2re=-Bl13*vb3re-Bl11*vb1re
vb2im=-Bl13*vb3im-Bl11*vb1im
#vb2=-Bl13*vb3re-1j*Bl13*vb3im-T*vb
#print vb2-vb2re-1j*vb2im
#print vb2re
#print vb2im
print np.abs(vb2re+1j*vb2im)/12.66e3

#print(nx.bfs_predecessors(G,0))
#nx.bfs_tree(G,0)
#print loopedges
#print treeedges

for iter in range(8):
    for itemload in loaddict:
        v=vb2re[loaddict[itemload][2]]+1j*vb2im[loaddict[itemload][2]]
        #print np.abs(v)
        il1re0[loaddict[itemload][2]]=np.real(np.conj((loaddict[itemload][0]+1j*loaddict[itemload][1])/(v)))
        il1im0[loaddict[itemload][2]]=np.imag(np.conj((loaddict[itemload][0]+1j*loaddict[itemload][1])/(v)))
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
    print res
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