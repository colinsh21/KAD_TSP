import networkx as nx
import itertools

#Testing network formation functions

def init_KAD_node(KAD, head_layer, label):
    # Add node with no preds

    # add knowledge or decision
    kd_label = KAD.number_of_nodes() + 1
    KAD.add_node(kd_label, layer=head_layer, label=label)

    return KAD

def add_KAD_node(KAD,label,tails,head_layer,refine=None):
    #Add action node and generating node
    #tails = [node_labels] go to actions
    #action goes to head = 'layer'
    #refine = node_label connections a tail node to head

    #add action
    a_label=KAD.number_of_nodes()+1
    KAD.add_node(a_label, layer='a')
    e_list=[(tail,a_label) for tail in tails]
    KAD.add_edges_from(e_list)

    #add knowledge or decision
    kd_label=KAD.number_of_nodes()+1
    KAD.add_node(kd_label, layer=head_layer, label=label)
    KAD.add_edge(a_label,kd_label)

    if refine:
        KAD.add_edge(refine,kd_label)

    return KAD

KAD=nx.DiGraph()
# init k
KAD=init_KAD_node(KAD,label='k1', head_layer='k')

#create k2
KAD=add_KAD_node(KAD,label='k2', head_layer='k', tails=['k1'])



KAD.add_node('k1',layer='k')
KAD.add_node('k2',layer='k')
KAD.add_node('k3',layer='k')
KAD.add_node('k4',layer='k')

#a layer
KAD.add_node('a1',layer='a')
KAD.add_node('a2',layer='a')
KAD.add_node('a3',layer='a')
KAD.add_node('a4',layer='a')

#d layer
KAD.add_node('d1',layer='d')

#edges
KAD.add_edges_from([('k1','a1'),('a1','k2')]) #A1
KAD.add_edges_from([('k1','a2'),('k2','a2'),('a2','k3')]) #A2
KAD.add_edge('k1','k3') #Refinement from A2
KAD.add_edges_from([('k3','a3'),('a3','d1')]) #A3 Making a decision
# KAD.add_edges_from([('d1','a4'),('k2','a4'),('a4','k4'),('k2','k4')]) #A4 Using a decision to refine k2
KAD.add_edges_from([('d1','a4'),('k1','a4'),('a4','k4')]) #A4 Using a decision
