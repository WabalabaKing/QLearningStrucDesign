import numpy as np
from anastruct import SystemElements
import copy

class Structure:
    def __init__(self,Nodes,Elements):
        self.Node = Nodes
        self.Element = Elements

def runFEA(Struct):
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
    u_star = 0.017*1
    rho = 1120
    A = np.pi*(0.2*10**-3)**2
    element_type = 1
    material_type = 1
    EAs = E*A
    EIs = E*1/2*np.pi*(0.2*10**-3)**2
    ######################################
    #Define FEA Model
    ss = SystemElements(EA=EAs, EI=EIs)
    for el in Ele:
        loc1c=Nodes[el[0]]
        loc2c=Nodes[el[1]]
        length = np.sqrt((loc1c[0]-loc2c[0])**2+(loc1c[1]-loc2c[1])**2)
        ss.add_element(location = [loc1c,loc2c])

    #Add support
    for n in Nodes:
        X = Nodes[n]
        if X[0]==0:
            ss.add_support_fixed(node_id=n)
    # Add loads.
    for n in Nodes:
        X = Nodes[n]
        if X[1]==0 and X[0]==2:
            ss.point_load(Fy=-8, node_id=n)
    ss.solve()
    # Get visual results.
    ss.show_displacement()
    SR = ss.get_node_displacements(node_id=0)
    Nmax = []
    for result in range(len(SR)):
        Nmax += [np.sqrt(SR[result]['ux']**2+SR[result]['uy']**2)]
    maxsig = max(Nmax)
    return maxsig

Ele = [[1,7],[7,9],[9,3],[3,1]]
Nodes = {7:[ 0.0, 0.0],1:[0.0, 0.2],3:[2.0, 0.2],9:[2.0, 0.0]}
S = Structure(Nodes,Ele)
blah = runFEA(S)