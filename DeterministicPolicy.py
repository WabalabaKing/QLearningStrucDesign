# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:14:49 2024

@author: wz10
"""
import numpy as np
from anastruct import SystemElements
import copy
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
    
    
###############################################################################
u_star = 0.014*1
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

Struc = Structure(Active,Ele)
uOld = runFEA(Struc,Fix,FNode,1)
Reward=(1E-4*(u_star-uOld)**(-3))/SturWeight(Struc,rho)
ObjHis = [Reward]
while Reward!=-1 or Inactive!=[]:
    AL  = []
    ReL = []
    for N in Inactive:
        for M in Ele:
            for O in Oper:
                StruTemp,lTemp = Actions(N,FNode,M,Struc,O)
                ele = copy.deepcopy(StruTemp.Element)
                for i in range(len(ele)):
                    ele[i].sort()
                dup =  not(has_dup(ele))
                if dup:
                    AL+=[[N,M,O]]
                    uNew = runFEA(StruTemp,Fix,FNode,0)
                    delW = lTemp*rho
                    ReL += [-10000*(u_star-uNew)**2-1000*delW]
    BestAction = AL[ReL.index(max(ReL))]
    Inactive.remove(BestAction[0])
    BestStruct,Ladded = Actions(BestAction[0],FNode,BestAction[1],Struc,BestAction[2])
    uI =  runFEA(BestStruct,Fix,FNode,0)
    Wnew = SturWeight(BestStruct,rho)
    RewardNew = (1E-4*(u_star-uI)**(-3))/Wnew
    if (u_star-uI)>=0:
        Reward = -1
        Struc = BestStruct
        s1 = runFEA(BestStruct,Fix,FNode,1)
        print('Local Optimum Found!, Displacement is '+str(uI)+' Weight is '+str(Wnew))
        break
    else:
        Struc = BestStruct
        Reward=RewardNew
        ObjHis+=[Reward]
        s1 = runFEA(BestStruct,Fix,FNode,1)
        print('Iterating, Displacement is '+str(s1)+' Weight is '+str(Wnew))
    Ele = copy.deepcopy(BestStruct.Element)
    uOld = uI
    