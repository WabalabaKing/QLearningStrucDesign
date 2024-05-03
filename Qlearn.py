# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:55:48 2024

@author: wz10
"""

import numpy as np
from anastruct import SystemElements
import copy
import random
import matplotlib.pyplot as plt
class Structure:
    def __init__(self,Nodes,Elements):
        self.Node = Nodes
        self.Element = Elements
def runFEA(Struct,Fix,Force,plot):
    Nodes = copy.deepcopy(Struct.Node)
    Ele = copy.deepcopy(Struct.Element)
    NNodes = {}
    NEle = [ [0]*2 for i in range(len(Ele))]
    #remesh
    counter = 1
    for n in Nodes:
        NNodes[counter] = Nodes[n]
        for i in range(len(Ele)):
            for j in range(len(Ele[0])):
                if Ele[i][j]==n:
                    NEle[i][j]=counter                        
        counter+=1
    Nodes = NNodes
    Ele = NEle
    E = 6.1*10**9
    A = np.pi*(0.2*10**-3)**2
    EAs = E*A
    EIs = E*1/2*np.pi*(0.2*10**-3)**2
    ######################################
    #Define FEA Model
    ss = SystemElements(EA=EAs, EI=EIs)
    for el in Ele:
        loc1c=Nodes[el[0]]
        loc2c=Nodes[el[1]]
        ss.add_element(location = [loc1c,loc2c])
    #Add support
    for n in Fix:
        ss.add_support_fixed(node_id=ss.find_node_id(n))
    # Add loads.
    for n in Force:
        ss.point_load(Fy=-8, node_id=ss.find_node_id( n))
    ss.solve()
    # Get visual results.
    if plot==1:
        ss.show_structure()
    else:
        pass
    SR = ss.get_node_displacements(node_id=0)
    Nmax = []
    for result in range(len(SR)):
        Nmax += [np.sqrt(SR[result]['ux']**2+SR[result]['uy']**2)]
    maxsig = max(Nmax)
    return maxsig

def CalcDist(Node1, Node2):
    x1,y1= Node1[0],Node1[1]
    x2,y2= Node2[0],Node2[1]
    dist = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist

    
    
    
def Actions(ANode,FNode, Member,Struc,Operator):
    addL = 0
    Nodes =copy.deepcopy( Struc.Node)
    Ele = copy.deepcopy(Struc.Element)
    #Nodes[int(ANode[0])] = [ANode[1],ANode[2]]
    Node2 =[ANode[1],ANode[2]]
    Node3 = FNode[0]
    
    if Operator =='T':
        Ele.remove(Member)
        addL-=CalcDist(Nodes[Member[0]], Nodes[Member[1]])
    #Does New Node lives on exisiting edges?
    IEle = []
    for el in Ele:
        Vec1 = np.array(Nodes[el[0]])-np.array(Nodes[el[1]])
        uVec1 = Vec1/np.linalg.norm(Vec1)
        Vec2 = np.array(Nodes[el[0]])-np.array(Node2)
        uVec2 = Vec2/np.linalg.norm(Vec2)
        if uVec1[0]==uVec2[0] and uVec1[1]==uVec2[1]:
            IEle+=[el]
    for el in IEle:
        Ele.remove(el)
        Ele+=[[el[0],ANode[0]]]
        Ele+=[[el[1],ANode[0]]]
    Ele+=[[ANode[0],Member[0]]]
    addL+=CalcDist(Node2, Nodes[Member[0]])
    Ele+=[[ANode[0],Member[1]]]
    addL+=CalcDist(Node2, Nodes[Member[1]])
    if Operator =='T':
        #Calculate nearestNode
        dis = []
        disID = []
        for n in Nodes:
            if Nodes[n]!=Nodes[Member[0]] and Nodes[n]!=Nodes[Member[1]]:
                Node1 = Nodes[n]
                d1 = CalcDist(Node1,Node2)
                d2 = CalcDist(Node1, Node3)
                disID+=[n]
                dis+=[d1+d2]
        mindis = dis.index(min(dis))
        N = disID[mindis]
        Ele+=[[ANode[0],N]]
        addL+=CalcDist(Node2, Nodes[N])
    Nodes[int(ANode[0])] = [ANode[1],ANode[2]]
    StrucNew = Structure(Nodes,Ele)
    return StrucNew, addL

def has_dup(seq):
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return len(seq) != len(unique_list)
   
    
def SturWeight(Struct,rho):
    Nodes = Struct.Node
    Ele = Struct.Element
    TotL = 0
    for M in Ele:
        Node1 = Nodes[M[0]]
        Node2 = Nodes[M[1]]
        TotL+=CalcDist(Node1, Node2)
    TotL = TotL*rho
    return TotL

def NicerElem(Struct):
    Ele = copy.deepcopy(Struct.Element)
    for i in range(len(Ele)):
        Ele[i].sort()
    Ele.sort()
    return Ele
###############################################################################
u_star = 0.016*1
rho = 1120*np.pi*(0.2*10**-3)**2
NodeLists = [[1,0.0,0.2],[2,1.0,0.2],[3,2.0,0.2],\
             [4,0.0,0.1],[5,1.0,0.1],[6,2.0,0.1],\
             [7,0.0,0.0],[8,1.0,0.0],[9,2.0,0.0]]

Ele = [[1,7],[7,9],[9,3],[3,1]]
Active = {7:[ 0.0, 0.0],1:[0.0, 0.2],3:[2.0, 0.2],9:[2.0, 0.0]}

Inactive = [             [2,1.0,0.2],\
             [4,0.0,0.1],[5,1.0,0.1],[6,2.0,0.1],\
                         [8,1.0,0.0]]
Fix = [[0.0,0.0],[0.0,0.2]]
FNode = [[2.0, 0.0]]
Oper = ["D","T"]
seedStru = Structure(Active,Ele)
###############################Define Q-learning Parameters####################
Qtable = [[[0],-np.infty,np.infty ]]   #Initialize Q_table
greedy = 0.0  # 0.2 percent chance of exploiting
EpisodeL = [400] #Play 100 episode
alpha = 0.8   #0.5 learn rate
reward = -np.infty
WsSD = []
count = 0
###############################Do Q-learning###################################
for Episode in EpisodeL:
    tempSD = []
    Qtable = [[[0],-np.infty,np.infty ]]   #Initialize Q_table
    for NN in range(Episode):
        greedy = (1/Episode)*NN
        Struc = copy.deepcopy(seedStru)
        useed= runFEA(Struc,Fix,FNode,0)
        Reward=(1E-4*(u_star-useed)**(-3))/SturWeight(Struc,rho)
        Inact = copy.deepcopy(Inactive)
        AEle = copy.deepcopy(Ele)
        ReH = 0
        while Reward!="Done" and Inact!=[]:
            niceElement = NicerElem(Struc)
            AL = []
            #Get a list of state-action pair
            for N in Inact:
                for M in AEle:
                    for O in Oper:
                        StruTemp,lTemp = Actions(N,FNode,M,Struc,O)
                        ele = copy.deepcopy(StruTemp.Element)
                        for i in range(len(ele)):
                            ele[i].sort()
                        dup =  not(has_dup(ele))
                        if dup:
                            AL+=[[niceElement,N,M,O]]
            Qk = list(list(zip(*Qtable))[0])
            Explore = []
            Exploit = []
            for SA in AL:
                if SA not in Qk:
                    Qtable+=[[SA,-99,0]]
                    Explore+=[1]
                else:
                    Exploit+=[1]
            #Explore or Exploit?
            flag = random.choices(['L','E'],weights = [(1-greedy),greedy],k = 1)[0]
            #If exploit, go find the biggest expected reward and play that action    
            if  flag =='E':
                Qr = list(list(zip(*Qtable))[1])
                Qk = list(list(zip(*Qtable))[0])
                Qrr = []
                Qkk = []
                for i in Qk:
                    if i[0]==NicerElem(Struc):
                        Qkk+=[i]
                        Qrr+=[Qr[Qk.index(i)]]
                SA = Qkk[Qrr.index(max(Qrr))]
                Na, Nm, No= SA[1],SA[2],SA[3]
                NewStru,Ladded = Actions(Na,FNode,Nm,Struc,No)
                uNew = runFEA(NewStru,Fix,FNode,0)
                delW = Ladded*rho
                Qtable[Qk.index(SA)][2] = Qtable[Qk.index(SA)][2]+1
                rewardN = -10000*(u_star-uNew)**2-1000*delW            
            if flag =='L':
                
                Qk = list(list(zip(*Qtable))[0])
                Qa = list(list(zip(*Qtable))[2])
                Qr = list(list(zip(*Qtable))[1])
                Qkk = []
                Qaa = []
                Qrr = []
                for i in Qk:
                    if i[0]==NicerElem(Struc):
                        Qkk+=[i]
                        Qaa+=[Qa[Qk.index(i)]]
                        Qrr+=[Qr[Qk.index(i)]]
                SA = Qkk[Qaa.index(min(Qaa))]
                Na, Nm, No= SA[1],SA[2],SA[3]
                NewStru,Ladded = Actions(Na,FNode,Nm,Struc,No)
                uNew = runFEA(NewStru,Fix,FNode,0)
                delW = Ladded*rho
                rewardN = -10000*(u_star-uNew)**2-1000*delW
                Qtable[Qk.index(SA)][2] = Qtable[Qk.index(SA)][2]+1
                if Qr[Qk.index(SA)]== 0:
                    Qtable[Qk.index(SA)][1] = rewardN
                else:
                    # get Q(st+1)
                    temp = []
                    for i in Qk:
                        if i[0] == NicerElem(NewStru):
                           temp+=[Qr[Qk.index(i)]]
                    try:             
                        Qt1 = max(temp)
                        Qtable[Qk.index(SA)][1] = (1-alpha)*Qr[Qk.index(SA)]+\
                                        (alpha)*(rewardN+Qt1)
                    except:
                        Qtable[Qk.index(SA)][1] = (1-alpha)*Qr[Qk.index(SA)]+\
                                        (alpha)*(rewardN-0)
                        
                        
            #Update Qbest
            
            ReH+=rewardN
            #is terminal state reached?
            Wnew = SturWeight(NewStru,rho)
            if (u_star-uNew)>=0:
                
                if NN%100==0:
                    print('termial state for episode ' +str(NN)+' Displacement is '+str(uNew) +' Weight is '+str(Wnew))
                Reward = 'Done'
                
            else:
                
                RewardN = (1E-4*(u_star-uNew)**(-3))/Wnew
                Inact.remove(Na)
                AEle = copy.deepcopy(NewStru.Element)
                Struc = copy.deepcopy(NewStru)
                Reward = RewardN
            count+=1
        tempSD+=[uNew/Wnew]
    WsSD+=[Wnew]
    s1 = runFEA(NewStru, Fix, FNode, 1)
plt.plot(EpisodeL, WsSD ,marker='.')
plt.ylabel('Weight of Resultant Structure [Kg]')
plt.xlabel('Episode')
plt.show()