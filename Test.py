import networkx as nx
import itertools
# Testing KAD Format and Flatten

KAD=nx.DiGraph()
#k layer
KAD.add_node('k1',layer='k',step=0)
KAD.add_node('k2',layer='k',step=1)
KAD.add_node('k3',layer='k',step=2)
KAD.add_node('k4',layer='k',step=4)

#a layer
KAD.add_node('a1',layer='a',step=1)
KAD.add_node('a2',layer='a',step=2)
KAD.add_node('a3',layer='a',step=3)
KAD.add_node('a4',layer='a',step=4)

#d layer
KAD.add_node('d1',layer='d',step=3)

#edges
KAD.add_edges_from([('k1','a1'),('a1','k2')]) #A1
KAD.add_edges_from([('k1','a2'),('k2','a2'),('a2','k3')]) #A2
KAD.add_edge('k1','k3') #Refinement from A2
KAD.add_edges_from([('k3','a3'),('a3','d1')]) #A3 Making a decision
# KAD.add_edges_from([('d1','a4'),('k2','a4'),('a4','k4'),('k2','k4')]) #A4 Using a decision to refine k2
KAD.add_edges_from([('d1','a4'),('k1','a4'),('a4','k4')]) #A4 Using a decision

#Flatten to KD
#iterate through nodes, if edge points to action->flatten, if edge points to otherwise->keep
KD=nx.DiGraph()
for i in KAD.nodes():
    if KAD.node[i]['layer']=='a':
        continue

    for j in KAD.successors(i):
        if KAD.node[j]['layer']=='a':
            #print n,tails
            heads=KAD.successors(j)
            #print n,e_list
            e_list=[(i,head) for head in heads]
            KD.add_edges_from(e_list)
        else:
            KD.add_edge(i,j)
print 'KD Edges 1:', KD.edges()
kd_1=KD.edges()

#copy and delete
KD = KAD.copy()
for i in KD.nodes():
    if KD.node[i]['layer'] == 'a':
        step=KD.node[i]['step']
        #get successors
        tails=KD.predecessors(i)

        #get head
        heads=KD.successors(i)

        #Add edges
        for tail in tails:
            for head in heads:
                KD.add_edge(tail,head,step=step)

        KD.remove_node(i)

print 'KD Nodes 2:', KD.nodes(data=True)
print 'KD Edges 2:', KD.edges()
kd_2=KD.edges()
print 'KD Equal?', set(kd_1)==set(kd_2)
#Flatten to A
#iterate through nodes, if edge points to action->flatten, if edge points to otherwise->keep
# P=nx.DiGraph()
# project_layers=['a']#['k','d']
# for i in KAD.nodes():
#     if KAD.node[i]['layer'] in project_layers:
#         continue
#     for j in KAD.successors(i):
#         if KAD.node[j]['layer'] in project_layers:
#             #print n,tails
#             heads=KAD.successors(j)
#             #print n,e_list
#             e_list=[]
#             for head in heads:
#                 if KAD.node[head]['layer'] not in project_layers: #ignore intra-layer connections for projected
#                     e_list.append((i,head))
#             #e_list=[(i,head) for head in heads]
#             print 'P', j, e_list
#             P.add_edges_from(e_list)
#         else:
#             P.add_edge(i,j)

P_t=nx.DiGraph()
project_layers=['a']
for i in KAD.nodes():
    relations = []
    for j in list(set(KAD.successors(i)+KAD.predecessors(i))):
        if KAD.node[j]['layer'] not in project_layers: #project i
            relations.append(j)

    e_list=list(itertools.combinations(relations,2))
    print 'Pt',i, e_list

    #e_list=[(i,head) for head in heads]
    P_t.add_edges_from(e_list)

A=nx.DiGraph()
project_layers=['k','d']
for i in KAD.nodes():
    if KAD.node[i]['layer'] in project_layers:
        continue

    for j in KAD.successors(i):
        if KAD.node[j]['layer'] in project_layers:
            #print n,tails
            heads=KAD.successors(j)
            #print n,e_list
            e_list=[(i,head) for head in heads]
            A.add_edges_from(e_list)
        else:
            A.add_edge(i,j)

#copy and delete A
A=KAD.copy()
for i in A.nodes():
    if A.node[i]['layer'] in ['k','d']:
        #get successors
        preds=A.predecessors(i)
        tails=[]
        for p in preds:
            if A.node[p]['layer'] not in ['k','d']:
                tails.append(p)

        #get head
        sucs=A.successors(i)
        heads = []
        for s in sucs:
            if A.node[s]['layer'] not in ['k', 'd']:
                heads.append(s)

        #Add edges
        for tail in tails:
            for head in heads:
                step = A.node[head]['step'] #step related to head
                A.add_edge(tail,head,step=step)

        #Add edges if initialized k
        if not tails: #no preds
            e_bad_direction=list(itertools.combinations(heads, 2)) #ID edges
            for e in e_bad_direction: #iterate through edgs
                if e[1]<e[0]: #Flip if reverse order
                    e=(e[1],e[0])
                step=A.node[e[0]]['step'] #step related to head
                print i, e
                A.add_edge(e[0],e[1],step=step)

        A.remove_node(i)
print 'A Nodes:', A.nodes(data=True)
print 'A Edges:', A.edges()

print 'KAD Nodes:', KAD.nodes(data=True)
print 'KAD Edges:', KAD.edges()

#print 'KD Nodes:', KD.nodes(data=True)
print 'KD Edges:', KD.edges()

#print 'P Nodes:', P.nodes(data=True)
# print 'P Edges:', P.edges()

#print 'Pt Nodes:', P_t.nodes(data=True)
#print 'Pt Edges:', P_t.edges()

print 'A Edges:', A.edges()
