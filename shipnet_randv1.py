# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:52:03 2015

@author: colinsh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 09:56:27 2015

@author: colinsh
"""

# Standard imports
import copy
import itertools

# Scientific computing imports
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from operator import itemgetter
from heapq import heappush, heappop
from itertools import count


class MultiPlex(object):
    def __init__(self):
        self.plex=[]
        self.name=[]
        self.global_node={}
        
    def add_plex(self,name):
        self.plex.append(nx.Graph())
        self.name.append(name)
        #print self.plex
        if len(self.plex)>1:
            for node in self.plex[-2].nodes():
                print node,self.plex[-2].nodes()
                if self.plex[-2].node[node]:
                    self.plex[-1].add_node(node,self.plex[-2].node(node))
                else:
                    self.plex[-1].add_node(node)
                
        
    def add_plex_node(self,node_name,data={}):        
        self.global_node[node_name]=data
        for plex in self.plex:
            if node_name in plex.nodes(): #node exists
                print 'node exists'
                break
            else:
                plex.add_node(node_name,data)
                
    def remove_plex_node(self,node_name):
        self.global_node.pop(node_name,None)
        for plex in self.plex:
            plex.remove_node(node_name)
                    
    def edit_plex_node(self,node_name,node_data):
        if node_name not in self.global_node:
            print 'node does not exist'
            return
        else:
            for plex in self.plex:
                plex.node[node_name]=node_data
                
            self.global_node[node_name]=node_data
    
    def add_plex_edge(self,(u,v,plex),data={}):
        #check if u and v exist
        if plex not in self.name:
            print 'plex does not exist'
            return
        elif (u not in self.global_node) or (v not in self.global_node):
            print u, 'or', v, 'does not exist'
            return
        else:
            plex_index=self.name.index(plex)
            self.plex[plex_index].add_edge(u,v,data)
        
                
    def plexes(self):
            return iter(self.plex)
    
    def names(self):
        return iter(self.name)
    
    def global_nodes(self):
        return iter(self.global_node)
                
 
    def __repr__(self):
        skip_none = True
        repr_string = type(self).__name__ + " ["
        except_list = "model"

        elements = [e for e in dir(self) if str(e) not in except_list]
        for e in elements:
            # Make sure we only display "public" fields; skip anything private (_*), that is a method/function, or that is a module.
            if not e.startswith("_") and eval('type(self.{0}).__name__'.format(e)) not in ['DataFrame', 'function', 'method', 'builtin_function_or_method', 'module', 'instancemethod']:
                    value = eval("self." + e)
                    if value != None and skip_none == True:
                        repr_string += "{0}={1}, ".format(e, value)

        # Clean up trailing space and comma.
        return repr_string.strip(" ").strip(",") + "]"
        
class ShipNET(object):
    def __init__(self,initial_grid_size):
        self.initial_grid_size=initial_grid_size
        self.ship=nx.grid_graph(dim=initial_grid_size)
        self.layout=nx.spectral_layout(self.ship)
        
        #label streets and penetrations
        self.dual_streets={}
        self.geometric_characteristics={'H':'horizontal','L':'longitudinal','T':'transverse'}
        structural_denotation=['T','L','H']
        self.transfer_types={'P':None,'EM_P':None,'C':None,'I':None,'H':None}
        #self.transfer_types={'E':None,'M':None,'I':None,'H':None}
        for u,v in self.ship.edges():
            self.ship[u][v]['availability']=list(self.transfer_types)
        
        for i in range(len(self.initial_grid_size)):
            for structure in range(self.initial_grid_size[i]+1):
                s_name='{}{}'.format(structural_denotation[i],structure)
                self.dual_streets[s_name]={}
                self.dual_streets[s_name]['orientation']=self.geometric_characteristics[structural_denotation[i]]
                self.dual_streets[s_name]['availability']=list(self.transfer_types)
                
        for u,v in self.ship.edges():
            coord_change=[(a is b) for a, b in zip(u,v)].index(False)
            if coord_change==0: #move in x
                self.ship[u][v]['penetration']='T{}'.format(max(u[0],v[0]))
                self.ship[u][v]['streets']=['L{}'.format(u[1]),'L{}'.format((u[1]+1)),'H{}'.format(u[2]),'H{}'.format((u[2]+1))]   

            if coord_change==1: #move in y
                self.ship[u][v]['penetration']='L{}'.format(max(u[1],v[1]))
                self.ship[u][v]['streets']=['H{}'.format(u[2]),'H{}'.format((u[2]+1)),'T{}'.format(u[0]),'T{}'.format((u[0]+1))]

            if coord_change==2: #move in z
                self.ship[u][v]['penetration']='H{}'.format(max(u[2],v[2]))
                self.ship[u][v]['streets']=['T{}'.format(u[0]),'T{}'.format((u[0]+1)),'L{}'.format(u[1]),'L{}'.format((u[1]+1))]

            #print (u,v),self.ship[u][v]['streets'],self.ship[u][v]['penetration']

        #generate default information dual
        self.ship_dual=nx.Graph()
        for key in self.dual_streets:
            if self.dual_streets[key]['orientation'] is 'horizontal':
                for cross_street in self.dual_streets:
                    if self.dual_streets[cross_street]['orientation'] is not 'horizontal':
                        self.ship_dual.add_edge(key,cross_street)

            if self.dual_streets[key]['orientation'] is 'transverse':
                for cross_street in self.dual_streets:
                    if self.dual_streets[cross_street]['orientation'] is not 'transverse':
                        self.ship_dual.add_edge(key,cross_street)

            if self.dual_streets[key]['orientation'] is 'longitudinal':
                for cross_street in self.dual_streets:
                    if self.dual_streets[cross_street]['orientation'] is not 'longitudinal':
                        self.ship_dual.add_edge(key,cross_street)
                        
    def remove_ship_node(self,node_name):
        self.ship.remove_node(node_name)
    
    def remove_ship_edge(self,edge_name):
        self.ship.remove_edge(edge_name)
    
    def relative_position(self,structure,node):
        if self.dual_streets[structure]['orientation'] is 'transverse':
            key_coord=0
        if self.dual_streets[structure]['orientation'] is 'longitudinal':
            key_coord=1
        if self.dual_streets[structure]['orientation'] is 'horizontal':
            key_coord=2
        
        #print key_coord
             
        coord_value=[]

        for u,v in self.ship.edges():
            if structure in self.ship[u][v]['streets']:
                coord_value=u[key_coord]
                #print coord_value
                break
        return node[key_coord]>coord_value

    def transfer_graphs(self):
        for transfer in self.transfer_types:
            s_g=nx.Graph()
            for j,k,d in self.ship.edges(data=True):
                available_streets=[]
                #print (j,k),d
                if transfer in self.ship[j][k]['availability']:
                    for street in d['streets']:
                        if transfer in self.dual_streets[street]['availability']:
                            available_streets.append(street)
                    if available_streets:
                        s_g.add_edge(j,k,streets=available_streets)
            self.transfer_types[transfer]=s_g
            
    def set_dc_config(self,water_tight_bulkheads,dc_deck_level,exception=[],p=0.0):
        self.num_bh=len(water_tight_bulkheads)
        self.height_dc=dc_deck_level
        self.permeability=p
        dc_deck='H{}'.format(self.height_dc)
        for u,v in self.ship.edges():
            #print (u,v), self.relative_position(dc_deck,u)
            #print (u,v),(self.ship[u][v]['penetration'] in water_tight_bulkheads) and (not self.relative_position(dc_deck,u))
            if (self.ship[u][v]['penetration'] in water_tight_bulkheads) and (not self.relative_position(dc_deck,u)):
                if numpy.random.rand(1)[0]>self.permeability: #if true, create watertight bh
                    self.ship[u][v]['availability']=exception
    
    def gen_disjoint_sets(self,affordance_multiplex,method,k=10):
        self.affordance_multiplex=affordance_multiplex
        self.transfer_graphs()
        self.disjoint_sets={}
        self.num_paths=k
        for plex,name in zip(self.affordance_multiplex.plexes(),self.affordance_multiplex.names()):
            for u,v in plex.edges():
                #print u,v,name
                #print u,self.affordance_multiplex.global_node[u]['loc']
                base_graph=self.transfer_types[name]
                n1=self.affordance_multiplex.global_node[u]['loc_possible'][-1]
                n2=self.affordance_multiplex.global_node[v]['loc_possible'][-1]
                if n1 not in self.disjoint_sets:
                    self.disjoint_sets[n1]={}
                if n2 not in self.disjoint_sets:
                    self.disjoint_sets[n2]={}
                if method=='shortest':
                    paths=nx.all_shortest_paths(base_graph, n1, n2)
                if method=='k-shortest':
                    k_paths=self.k_shortest_paths(base_graph, n1, n2, k=self.num_paths, weight='weight')
                    paths = [list(x) for x in set(tuple(x) for x in k_paths[1])]
                step_paths=[]
                for p in paths:
                    step_paths.append(p[1:-1])
                disjoint_paths=[]
                disjoint_paths_rev=[]
                cutoff=float("inf")
                if len(step_paths)>cutoff:
                    step_paths=itemgetter(*numpy.random.choice(range(len(step_paths)), cutoff, replace=False))(step_paths)
                for d_set in self.comb(step_paths):
                    d_set_rev=[]
                    for path_index in xrange(len(d_set)):
                        if not d_set[path_index]:
                            d_set[path_index].insert(0,n1)
                            d_set[path_index].append(n2)
                            if n1==n1:
                                d_set[path_index]=d_set[path_index][1:-1]
                        elif d_set[path_index][0]!=n1: #check if head and tail are appended
                            d_set[path_index].insert(0,n1)
                            d_set[path_index].append(n2)
                            if n1==n2:
                                d_set[path_index]=d_set[path_index][1:-1]
                    disjoint_paths.append(d_set)
                    for p in d_set:
                        d_set_rev.append(p[::-1])
                    disjoint_paths_rev.append(d_set_rev)
                
                self.disjoint_sets[n1][n2]=disjoint_paths
                self.disjoint_sets[n2][n1]=disjoint_paths_rev
    
    def affordance_routing(self,affordance_multiplex,num_arrangements,method='random',wander=1.5,redundancy=1):
        self.affordance_multiplex=affordance_multiplex
        self.num_arrangements=num_arrangements
        self.wander=wander
        self.redundancy=redundancy
        self.method=method
        #first get the available subgraph
        self.transfer_graphs()
            
        #create dataframe for results
        self.affordance_edges=[]
        for plex,name in zip(self.affordance_multiplex.plexes(),self.affordance_multiplex.names()):
            #print name
            self.affordance_nodes=plex.nodes()
            for u,v in plex.edges():
                #print edge,name
                self.affordance_edges.append((u,v,name))
        
        #print 'edges',self.affordance_edges
                
        self.arr_e_list_method=[]
        self.arr_n_list_method=[]

        fixed_loc=pd.DataFrame(index=self.ship.nodes(),columns=self.affordance_nodes)
        fixed_loc=fixed_loc.fillna(0.0)
        
        for n in self.affordance_multiplex.global_nodes():
            n_loc=self.affordance_multiplex.global_node[n]['loc_possible'][-1]
            fixed_loc.ix[n_loc,n]+=1
        
        #Set up disjoint sets
        #self.disjoint_sets(self.affordance_multiplex,self.method)

        #Arrangements
        arrangements=0
        while arrangements<self.num_arrangements:
            #print 'arrangement',arrangements
            arrangements+=1
            
            e_route=pd.DataFrame(index=self.ship.edges(),columns=self.affordance_edges)
            e_route=e_route.fillna(0.0)
            #print self.e_route

            n_location=pd.DataFrame(index=self.ship.nodes(),columns=self.affordance_nodes)
            n_location=n_location.fillna(0.0)
            #print self.n_location

            for n in self.affordance_multiplex.global_nodes():
                data=self.affordance_multiplex.global_node[n]
                n_loc=data['loc_possible'][-1]
                #print n_loc
                data['loc'].append(n_loc)
                #print data
                self.affordance_multiplex.edit_plex_node(n,data)
                n_location.ix[n_loc,n]+=1
                
            n_location['total']=n_location.sum(1)
            self.arr_n_list_method.append(n_location)
                                      
            for plex,name in zip(self.affordance_multiplex.plexes(),self.affordance_multiplex.names()):
                for u,v in plex.edges():
                    source=self.affordance_multiplex.global_node[u]['loc'][-1]
                    target=self.affordance_multiplex.global_node[v]['loc'][-1]
                        
                    e_list=[]
                    #print source, target
                    disjoint_paths=self.disjoint_sets[source][target]
                    max_redundancy=len(max(disjoint_paths,key=len))
                    if self.redundancy>max_redundancy:
                        poss_path_sets=filter(lambda x: len(x)==max_redundancy,disjoint_paths)
                    else:
                        poss_path_sets=filter(lambda x: len(x)==self.redundancy,disjoint_paths)
                    path_sets=poss_path_sets[numpy.random.randint(len(poss_path_sets))]
                    for path in path_sets:
                        e_list_t=[]
                        for i in range(len(path)-1):
                            e=(path[i],path[i+1])

                            if not e:
                                continue
                            elif e in e_route.index:
                                self.e_att=(e,(u,v,name))
                                e_route.ix[e,(u,v,name)] += 1
                                e_list_t.append(e)

                            else:
                                self.e_att=(e[::-1],(u,v,name))
                                e_route.ix[e[::-1],(u,v,name)] += 1
                                e_list_t.append(e[::-1])
                        e_list.append(e_list_t)
                    plex[u][v]['paths'].append(e_list)
            
            e_route['total']=e_route.sum(1)
            #get types on edges
            e_route['num_types']=numpy.nan
            for e in e_route.index:
                types=[]
                for a in e_route.columns:
                    if a=='total':
                        continue
                    if (e_route[a][e]>0) and (a[2] not in types):
                        types.append(a[2])
                e_route['num_types'][e]=len(types)
            
            e_route['c_f']=e_route.apply(lambda row: 2**row['total'],axis=1)                
            e_route['c_f_t']=e_route.apply(lambda row: 2**row['num_types'],axis=1)
            self.arr_e_list_method.append(e_route)
            
        return self.arr_n_list_method, self.arr_e_list_method
    
    def k_shortest_paths(self,G, source, target, k=1, weight='weight'):
    
        if source == target:
            return ([0], [[source]]) 

        length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
        if target not in length:
            raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))

        lengths = [length[target]]
        paths = [path[target]]
        c = count()        
        B = []                        
        G_original = G.copy()    

        for i in range(1, k):
            for j in range(len(paths[-1]) - 1):            
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]

                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G.has_edge(u, v):
                            edge_attr = G.edge[u][v]
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))

                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    # out-edges
                    for u, v, edge_attr in G.edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

                    if G.is_directed():
                        # in-edges
                        for u, v, edge_attr in G.in_edges_iter(node, data=True):
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))

                spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)            
                if target in spur_path and spur_path[target]:
                    total_path = root_path[:-1] + spur_path[target]
                    total_path_length = self.get_path_length(G_original, root_path, weight) + spur_path_length[target]                
                    heappush(B, (total_path_length, next(c), total_path))

                for e in edges_removed:
                    u, v, edge_attr = e
                    G.add_edge(u, v, edge_attr)

            if B:
                (l, _, p) = heappop(B)        
                lengths.append(l)
                paths.append(p)
            else:
                break

        return (lengths, paths)
 
    def get_path_length(self,G, path, weight='weight'):
        length = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]

                length += G.edge[u][v].get(weight, 1)

        return length
    
    def comb(self,input, lst = [], lset = set()):
        if lst:
            yield lst
        for i, el in enumerate(input):
            if lset.isdisjoint(el):
                for out in self.comb(input[i+1:], lst + [el], lset | set(el)):
                    yield out

    def get_incidence(self, n_locations,e_routes):
        
        self.arr_a_list=[]
        self.arr_ae_list=[]
        
        for arr in range(len(n_locations)): 
        
            a_incidence=pd.DataFrame(index=self.affordance_nodes,columns=self.affordance_nodes)
            a_incidence=a_incidence.fillna(0.0)
            #print self.a_incidence

            ae_incidence=pd.DataFrame(index=self.affordance_edges,columns=self.affordance_edges)
            ae_incidence=ae_incidence.fillna(0)
            #print self.ae_incidence
            #post-process locations
            for n1 in self.affordance_multiplex.global_nodes(): #get affordance incidence
                for n2 in self.affordance_multiplex.global_nodes():
                    #if n1==n2:
                        #continue
                    if n_locations[arr].index[n_locations[arr][n1]==1]==n_locations[arr].index[n_locations[arr][n2]==1]:
                        a_incidence.ix[n1,n2]+=1
                        #a_incidence.ix[n2,n1]+=1
            a_incidence['total']=a_incidence.sum(1)
            self.arr_a_list.append(a_incidence)
            
            #post-process edges
            for e1 in e_routes[arr].columns:
                length=e_routes[arr][e1].sum()
                #print length
                for i in e_routes[arr].index[e_routes[arr][e1]==1]:
                    for e2 in e_routes[arr].columns:
                        #print e_routes[-1][e2][i]
                        if isinstance(e1, str) or isinstance(e2, str): #e1==('total' or 'c_f') or e2==('total' or 'c_f'):
                            continue
                        if e1==e2:
                            continue
                        if e_routes[arr][e2][i]==1:
                            ae_incidence.ix[e1,e2]+=1.0#/length
                       
            ae_incidence['total']=ae_incidence.sum(1)
            ae_incidence['diff_total']=0.0
            #print ae_incidence
            for i in ae_incidence.index:
                diff_total=0
                for c in ae_incidence.columns:
                    if isinstance(c, str): #c==('total' or 'diff_total' or 'c_f'):
                        continue
                    if i[2] != c[2]:
                        #print i,c,ae_incidence[c][i]
                        diff_total+=float(ae_incidence[c][i])
                ae_incidence['diff_total'][i]=diff_total
            self.arr_ae_list.append(ae_incidence)
            
        return self.arr_a_list, self.arr_ae_list
    
    def node_complexity(self,n_location,e_route):
        self.arr_n_cmplx_list=[]
        for arr in range(len(n_location)):
            n_cmplx=pd.DataFrame(index=n_location[arr].index,columns=['c_a','c_e','c_e_t','c_f','c_f_t','p'])
            n_cmplx=n_cmplx.fillna(0.0)
            for n in n_location[arr].index:
                types=[]
                #print n
                #print n_location[arr]['total'][n]
                c_a=2.0**n_location[arr]['total'][n]
                #print n_cmplx['c_a'][n]
                c_e=0.0+self.redundancy*n_location[arr]['total'][n]
                for a in e_route[arr].columns:
                    if type(a) is tuple:
                        path=e_route[arr].index[e_route[arr][a]==1]
                        for e in path:
                            if n in e:
                                if a[2] not in types:
                                    types.append(a[2])
                                c_e+=1
                                break
                #print c_e
                if len(types)==8.0:
                    print types
                n_cmplx['c_a'][n]=c_a
                n_cmplx['c_e'][n]=c_e
                n_cmplx['c_e_t'][n]=len(types)
                n_cmplx['c_f'][n]=c_a*2.0**c_e
                n_cmplx['c_f_t'][n]=c_a*2.0**len(types)
                n_cmplx['p'][n]=2.0**(-(c_e-c_a))
               
            self.arr_n_cmplx_list.append(n_cmplx)
            
        return self.arr_n_cmplx_list
    
    def affordance_complexity(self,a_incidence,ae_incidence):
        self.arr_a_cmplx_list=[]
        for arr in range(len(a_incidence)):
            #print a_incidence[arr]
            #print a_incidence[arr].index
            a_cmplx=pd.DataFrame(index=a_incidence[arr].index,columns=['c_a','c_e','c_d','p'])
            a_cmplx=a_cmplx.fillna(0.0)
            for a in a_incidence[arr].index:
                c_a=2.0**float(a_incidence[arr]['total'][a])
                c_e=0.0
                c_e_d=0.0
                for ae in ae_incidence[arr].index:
                    if a in ae: #is affrodance in the connection
                        #print ae_incidence[arr]['total'][ae]
                        c_e+=float(ae_incidence[arr]['total'][ae])
                        #print c_e
                        c_e_d+=float(ae_incidence[arr]['diff_total'][ae])
                #c_f=c_a*2.0**c_e
                p=2.0**(-(c_e_d-c_a))
                #print a
                #print float(c_a)
                #print float(c_e)
                #print float(c_f)
                #print p
                #print a_cmplx
                #print 'c_e test',c_e
                a_cmplx['c_a'][a]=float(c_a)
                a_cmplx['c_e'][a]=float(c_e)
                #print a_cmplx['c_e'][a]
                a_cmplx['c_d'][a]=float(c_e_d)
                a_cmplx['p'][a]=p
                
            self.arr_a_cmplx_list.append(a_cmplx)
        return self.arr_a_cmplx_list
    
    def total_interactions(self,node_complx_table,e_routes): #input is output from node_complexity function
        self.total_int_list=[]
        for arr in range(len(node_complx_table)):
            types=[node_complx_table[arr]['c_f_t'].sum(0)+e_routes[arr]['c_f_t'].sum(0),
                    node_complx_table[arr]['c_f_t'].sum(0),
                    e_routes[arr]['c_f_t'].sum(0)]
            
            totals=[node_complx_table[arr]['c_f'].sum(0)+e_routes[arr]['c_f'].sum(0),
                    node_complx_table[arr]['c_f'].sum(0),
                    e_routes[arr]['c_f'].sum(0)]
            
            self.total_int_list.append((types,totals))
                #node_complx_table[arr]['c_f_t'].sum(0)+e_routes[arr]['c_f_t'].sum(0)                        
                #+e_routes[arr]['c_f'].sum(0))
                                        #node_complx_table[arr]['c_f'].sum(0)))
            #((node_complx_table[arr]['c_f'].sum(0),node_complx_table[arr]['c_f_t'].sum(0)),
                                         #(e_routes[arr]['c_f'].sum(0),e_routes[arr]['c_f_t'].sum(0)))
            
        return self.total_int_list
                
        
            
            
        
        
            
            
        