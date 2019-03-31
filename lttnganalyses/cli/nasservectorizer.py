# The MIT License (MIT)
#
# Copyright (C) 2015 - Julien Desfossez <jdesfossez@efficios.com>
#               2015 - Antoine Busque <abusque@efficios.com>
#               2015 - Philippe Proulx <pproulx@efficios.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author 2018 - Seyed Vahid Azhari <azharivs@gmail.com>
# Order of features in .vector files:
# Timer/Disk/Net/Task/Unknown/NonRoot/Root/L0_Preemption
#
# sample command line options 
# --top 3 --feature fti,fdi,fta,fne,wti,wdi,wta,wne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8
# --top 5 --feature fti,fdi,fta,fne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8
# --top 5 --feature wti,wdi,wta,wne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8

#TODO remove unnecessary LAMI baggage 
#TODO do a clustering for random data and use for comparison
#TODO consider distribution of similarity score among same class and other class (should be tightly packed)
#TODO highlight badly clustered samples: those with small intraclass similarity and those with large interclass similarity
#     E.g., sample : lowest three same class similarity scores with samples i1,i2,i3 (maximum of the three scores = same_cl)
#                    largest three overlaps with clusters j1,j2,j3 at samples k1,k2,k3 (maximum of the three scores = other_cl)
#                    where badly clustered samples are those with same_cl < average same cluster score or other_cl > average other cluster score
import operator
from ..common import format_utils
from .command import Command
from ..core import vectorizer
from . import mi
from . import termgraph
import math
import os

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from enum import Enum
import numpy as np
from scipy import sparse
import functools
import matplotlib.pyplot as plt
from matplotlib import cm

from igraph import load
from igraph import plot
from igraph import ClusterColoringPalette
from igraph import RainbowPalette
from igraph import drawing

import cairo

colors = ['#000000', '#15b01a', '#0343df', '#9a0eea', '#e50000', '#ff796c', '#ffff14', '#f97306', '#00ffff', '#01ff07', '#75bbfd']

#TODO do normalization over a column as well as over a row
# Row normalization compares a VMPID with itself, e.g., (Timer=300,Net=30) 
# will be the same as (Timer=30,Net=3), where Timer triggers 10 times more frequently than Net.
# Column-wise normalization would overcome this problem provided that workloads have the same
# time scale. That is, both are run for the same amount of time, otherwise the one running longer
# would definitely have larger absolute frequency. 
# One remedy is to use relative frequency (or rate) instead of absolute.
# For example, we can divide frequency by total execution time in root or non root.
# I will add a --rate option for this 

#cosine similarity matrix among samples
#input: samples array[n_samples, n_features]
#output: sim_matrix [n_samples, n_samples] 
def cosine_sim(samples):
    return cosine_similarity(samples, dense_output = True)
    
#compute euclidean similarity matrix and return in numpy array   
def euclidean_sim(samples):
    print("euclidean_sim_begin:",samples.shape[0])
    n_samples = samples.shape[0]
    d = np.zeros((n_samples,n_samples))
    return d #sva added to speed things up
    d_min = 0
    d_max = 0
    for i in range(n_samples):
        print("processing row ",i)
        for j in range(i,n_samples):
            d[i][j] = (sqdist(samples.toarray()[i],samples.toarray()[j]))**(1/2)
            d[j][i] = d[i][j]
            if d[i][j] < d_min:
                d_min = d[i][j]
            if d[i][j] > d_max:
                d_max = d[i][j]
    
    print("normalizing distances")        
    for i in range(n_samples):
        for j in range(n_samples):
            d[i][j] = 1-(d[i][j]-d_min)/(d_max-d_min)
    print("euclidean_sim_end")
    return d

#reorder euclidean similarity matrix with indexes sorted according to order
def reorder_sim(sim, order):
    print("reorder_sim_begin (with indexes sorted according to order)")
    n_samples = sim.shape[0]
    d = np.zeros((n_samples,n_samples))
    return d #sva added to speedup
    d_min = 0
    d_max = 0
    for i in range(n_samples):
        for j in range(i,n_samples):
            d[i][j] = sim[order[i]][order[j]]
            d[j][i] = d[i][j]
    print("reorder_sim_end (with indexes sorted according to order)")
    return d

#reorder and filter euclidean similarity matrix with indexes sorted according to order
def reorder_sim_filter(sim, order):
    print("reorder_sim_filter_begin (with indexes sorted according to order)")
    n_samples = len(order)
    d = np.zeros((n_samples,n_samples))
    d_min = 0
    d_max = 0
    for i in range(n_samples):
        for j in range(i,n_samples):
            d[i][j] = sim[order[i]][order[j]]
            d[j][i] = d[i][j]
    print("reorder_sim_filter_end (with indexes sorted according to order)")
    return d

#TODO Graph visualization
def graph_viz(samples, similarity, vertex_label,
    threshold, edge_label=None, 
    vertex_color=None, edge_color=None, 
    weight=True, bipartite=False):
    
    if not bipartite: #each vertex is a VMPID
        print('dfg')
        sim = similarity(samples)
        sim[sim < threshold] = 0 #set entries with low similarity to zero
        
    else: #bipartite graph of VMPIDs and Features
        print('dsdaffg')
    
    return 

#compute squared euclidean distance between two arrays
#sample is a numpy array
def sqdist(sample,centroid):
    return functools.reduce(lambda x,y:x+y, (sample-centroid)**2)

#TODO needs cleanup
#FIXME unable to graphically identify various traces when number is very large
def show_sim_matrix_proc_vm(proc_vm_label, samples_vm, cl_vm, vm_list, samples_proc, cl_proc, vmpid_list):
    print("show_sim_matrix_proc_vm")
    #sort samples with respect to labels
    sil = cl_proc[1]['total silhouette']
    name = 'KMEANS_Proc{}_VM{}'.format(max(cl_proc[0])+1,max(cl_vm[0])+1) #TODO generalize for other clusterings besides KMEANS
    labels = cl_proc[0]
    order = np.argsort(labels).tolist() 
    #compute similarity matrix
    d = reorder_sim(samples_proc, order)
    new_vmpid_list = list(vmpid_list)     
    print("show_sim_matrix_proc_vm_begin")
    new_vmpid_list = \
        [vmpid_list[i]+'['+str(labels[i])+','+str(proc_vm_label[vmpid_list[i][0:vmpid_list[i].rfind('/')]])+']'\
        for i in order] #concatenate cluster labels to vmpid name
    #plot it
    fig, ax = plt.subplots()
    cax = ax.imshow(d, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title(name+'\n'+'Similarity Matrix'+' (Silhouette = '+str(sil)+')')
    ax.set_yticks(np.arange(len(new_vmpid_list)))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticklabels(new_vmpid_list)
    ax.set_xticklabels(sorted(labels))
    #axr = ax.twinx()
    #axr.set_yticks(np.arange(len(labels)))
    #axr.set_yticklabels(sorted(labels))
    #axr.imshow(d, interpolation='nearest', cmap=cm.coolwarm)

    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['< 0', '0.25', '0.5', '0.75', '> 1'])  # vertically oriented colorbar
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.ion() #turn on interactive mode so execution does not block on show()
    plt.savefig("/home/azhari/temp2/"+name+".png", dpi=150, bbox_inches='tight')
    plt.show()
    #from io import BytesIO
    
    return 
    

#FIXME unable to graphically identify various traces when number is very large
#TODO refactor: too much stuff is done here
def show_sim_matrix(sim, vmpid_list, name, samples, cl, legend = None, short_name = False, title = False, show = False):
    print("show_sim_matrix")
    arrsamples = samples.toarray()
    labels = cl[0]
    centroids = cl[1]['centroids']
    sil = cl[1]['total silhouette']
    c_sil = cl[1]['cluster silhouette']
    c_stderr = cl[1]['cluster stderr']
    print("#sort samples with respect to labels ...")
    order = np.argsort(labels).tolist() 
    print("#compute similarity matrix ...")
    d = reorder_sim(sim, order)
    if short_name:
        vmpid_list = [vmpid_list[i].split(':')[0][0:-4]+'['+str(labels[i])+']' for i in order] #concatenate shorthanded cluster labels to vmpid name
    else:
        vmpid_list = [vmpid_list[i]+'['+str(labels[i])+']' for i in order] #concatenate cluster labels to vmpid name
    #plot it
    print("start plot...")
    fig, ax = plt.subplots()
    cax = ax.imshow(d, interpolation='nearest', cmap=cm.coolwarm)
#    if title:
#        ax.set_title(name+'\n'+'Similarity Matrix'+' (Silhouette = '+str(sil)+')')
#    ax.set_yticks(np.arange(len(vmpid_list)))
#    ax.set_xticks(np.arange(len(labels)))
#    ax.set_yticklabels(vmpid_list, fontdict = {'fontsize':8})
#    ax.set_xticklabels(sorted(labels), fontdict = {'fontsize':10})
    #axr = ax.twinx()
    #axr.set_yticks(np.arange(len(labels)))
    #axr.set_yticklabels(sorted(labels))
    #axr.imshow(d, interpolation='nearest', cmap=cm.coolwarm)

    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #cbar = fig.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    #cbar.ax.set_yticklabels(['< 0', '0.25', '0.5', '0.75', '> 1'])  # vertically oriented colorbar
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.ion() #turn on interactive mode so execution does not block on show()
#    plt.savefig("/home/azhari/temp2/"+name+".png", dpi=150, bbox_inches='tight')
#    if show:
#        plt.show()
    #also output centroids to text file -------------------------
    with open("/home/azhari/temp2/"+name+".centroids",'w') as centF:
        if legend != None:
            line = '          \t'
            for l in legend:
                line = line + l + '\t\t'
            centF.write(line+'\n')
        for i in range(centroids.shape[0]):
            line = 'Cluster #{}\t'.format(i)
            for j in range(centroids.shape[1]):
                line = line + "{0:.3f}\t\t".format(centroids[i][j])
            line = line + '\n'
            centF.write(line)
    
    #------------- centroids stacked bar chart -------------------------------
    print("------------- centroids stacked bar chart -------------------------------")
    plt.subplots()
    p = []
    ind = np.arange(centroids.shape[0])
    lft = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[1]): #loop column wise over features
        tmp = plt.barh(ind, centroids[:,i], height = 0.35, align = 'center', left = lft, color = colors[i])
        lft = lft + centroids[:,i]
        #print(lft)
        p.append(tmp[0])
    plt.xlabel('Normalized Usage')
    if title:
        plt.title('VM Prototype Usage Patterns '+name)
    plt.yticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
    plt.xticks(np.arange(0,3,0.5))
    if legend != None:
        plt.legend(p, legend)    
    plt.savefig("/home/azhari/temp2/"+name+"_centroids.png", dpi=150, bbox_inches='tight')
    
    #------------- clusters silhouette bar chart -------------------------------
    print("------------- clusters silhouette bar chart -------------------------------")
    plt.subplots()
    p = []
    ind = np.arange(centroids.shape[0])
    tmp = plt.bar(ind, c_sil, width = 0.35, align = 'center', color = colors[i])
    p.append(tmp[0])
    if title:
        plt.title('Cluster Silhouette '+name)
    plt.xticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
    plt.yticks(np.arange(0,1.1,0.1))
    #if legend != None:
    #    plt.legend(p, legend)    
    plt.savefig("/home/azhari/temp2/"+name+"_sil.png", dpi=150, bbox_inches='tight')
    
    #------------- clusters standard error bar chart -------------------------------
    print("------------- clusters standard error bar chart -------------------------------")
    plt.subplots()
    p = []
    ind = np.arange(centroids.shape[0])
    tmp = plt.bar(ind, c_stderr, width = 0.35, align = 'center', color = colors[i])
    p.append(tmp[0])
    if title:
        plt.title('Cluster Standard Error '+name)
    plt.xticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
    plt.yticks(np.arange(0,1.1,0.1))
    #if legend != None:
    #    plt.legend(p, legend)    
    plt.savefig("/home/azhari/temp2/"+name+"_stderr.png", dpi=150, bbox_inches='tight')
    
    #------------------ clusters membership text file --------------------------------            
    with open("/home/azhari/temp2/"+name+".clusters",'w') as clF:
        if legend != None:
            line = ''
            for l in legend:
                line = line + l + '\t\t'
            clF.write(line+'\n')
        for i in range(samples.shape[0]):
            line = ''
            for j in range(samples.shape[1]):
                line = line + "{0:.3f}\t\t".format(arrsamples[order[i]][j])
            line = line + vmpid_list[i]+'\n'
            clF.write(line)
    return 

#filtered by cl_filter clustering
#one plot of cl sim matrix per samples in same cluster of cl_filter    
def show_sim_matrix_filter(sim, vmpid_list_orig, name, samples, cl, cl_filter, legend = None):
    print("show_sim_matrix_filtered")
    arrsamples = samples.toarray()
    
    labels_ = cl[0]
    centroids = cl[1]['centroids']
    sil = cl[1]['total silhouette']
    c_sil = cl[1]['cluster silhouette']
    c_stderr = cl[1]['cluster stderr']
    
    labels_filter = cl_filter[0]
    #sort samples with respect to labels
    order_ = np.argsort(labels_).tolist() 
    for c_filter in range(cl_filter[1]['centroids'].shape[0]):
        print(c_filter)
        order = [order_[i] for i in range(len(order_)) if labels_filter[order_[i]]==c_filter]
        print(order)
        #compute similarity matrix
        d = reorder_sim_filter(sim, order)        
        vmpid_list = [vmpid_list_orig[i]+'['+str(labels_[i])+']' for i in order] #concatenate cluster labels to vmpid name
        labels = [labels_[i] for i in order]
        #plot it
        print("start plot...")
        fig, ax = plt.subplots()
        cax = ax.imshow(d, interpolation='nearest', cmap=cm.coolwarm)
        ax.set_title(name+' Filtered by C{}\n'.format(c_filter)+'Similarity Matrix'+' (Silhouette = '+str(sil)+')')
        ax.set_yticks(np.arange(len(vmpid_list)))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticklabels(vmpid_list)
        ax.set_xticklabels(sorted(labels))
        #axr = ax.twinx()
        #axr.set_yticks(np.arange(len(labels)))
        #axr.set_yticklabels(sorted(labels))
        #axr.imshow(d, interpolation='nearest', cmap=cm.coolwarm)
    
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
        cbar.ax.set_yticklabels(['< 0', '0.25', '0.5', '0.75', '> 1'])  # vertically oriented colorbar
        #fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #plt.ion() #turn on interactive mode so execution does not block on show()
        plt.savefig("/home/azhari/temp2/"+name+"_filter_C{}.png".format(c_filter), dpi=150, bbox_inches='tight')
        ##plt.show()
        #also output centroids to text file -------------------------
        with open("/home/azhari/temp2/"+name+"_filter_C{}.centroids".format(c_filter),'w') as centF:
            if legend != None:
                line = '          \t'
                for l in legend:
                    line = line + l + '\t\t'
                centF.write(line+'\n')
            for i in range(centroids.shape[0]):
                line = 'Cluster #{}\t'.format(i)
                for j in range(centroids.shape[1]):
                    line = line + "{0:.3f}\t\t".format(centroids[i][j])
                line = line + '\n'
                centF.write(line)
        
        #------------- centroids stacked bar chart -------------------------------
        print("------------- centroids stacked bar chart -------------------------------")
        plt.subplots()
        p = []
        ind = np.arange(centroids.shape[0])
        lft = np.zeros(centroids.shape[0])
        for i in range(centroids.shape[1]): #loop column wise over features
            tmp = plt.barh(ind, centroids[:,i], height = 0.35, align = 'center', left = lft, color = colors[i])
            lft = lft + centroids[:,i]
            #print(lft)
            p.append(tmp[0])
        plt.xlabel('Normalized Usage')
        plt.title('VM Prototype Usage Patterns '+name)
        plt.yticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
        plt.xticks(np.arange(0,3,0.5))
        if legend != None:
            plt.legend(p, legend)    
        plt.savefig("/home/azhari/temp2/"+name+"_centroids_filter_C{}.png".format(c_filter), dpi=150, bbox_inches='tight')
        
        #------------- clusters silhouette bar chart -------------------------------
        print("------------- clusters silhouette bar chart -------------------------------")
        plt.subplots()
        p = []
        ind = np.arange(centroids.shape[0])
        tmp = plt.bar(ind, c_sil, width = 0.35, align = 'center', color = colors[i])
        p.append(tmp[0])
        plt.title('Cluster Silhouette '+name)
        plt.xticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
        #plt.xticks(np.arange(0,3,0.5))
        if legend != None:
            plt.legend(p, legend)    
        plt.savefig("/home/azhari/temp2/"+name+"_sil_filter_C{}.png".format(c_filter), dpi=150, bbox_inches='tight')
        
        #------------- clusters standard error bar chart -------------------------------
        print("------------- clusters standard error bar chart -------------------------------")
        plt.subplots()
        p = []
        ind = np.arange(centroids.shape[0])
        tmp = plt.bar(ind, c_stderr, width = 0.35, align = 'center', color = colors[i])
        p.append(tmp[0])
        plt.title('Cluster Standard Error '+name)
        plt.xticks(ind, ['C{}'.format(i) for i in range(centroids.shape[0])])
        #plt.xticks(np.arange(0,3,0.5))
        if legend != None:
            plt.legend(p, legend)    
        plt.savefig("/home/azhari/temp2/"+name+"_stderr_filter_C{}.png".format(c_filter), dpi=150, bbox_inches='tight')
            
        #------------------ clusters membership text file --------------------------------            
        print("------------------ clusters membership text file --------------------------------")
        with open("/home/azhari/temp2/"+name+"_filter_C{}.clusters".format(c_filter),'w') as clF:
            if legend != None:
                line = ''
                for l in legend:
                    line = line + l + '\t\t'
                clF.write(line+'\n')
            for i in range(len(order)):
                line = ''
                for j in range(samples.shape[1]):
                    line = line + "{0:.3f}\t\t".format(arrsamples[order[i]][j])
                line = line + vmpid_list[i]+'\n'
                clF.write(line)
    return 

#performs kmeans clustering on a list of samples
#inparams: a dict of input parameters
def kmeans_clustering(samples, inparams):
    print("kmeans_clustering_begin",samples.shape[0])
    n = min (inparams['n_clusters'], samples.shape[0]) #number of clusters never larger than number of samples
    cl = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=100, random_state=None)
    print("Fitting samples ...")    
    cl.fit(samples)
    print("Computing sqare root of distance to centroids...")    
    d = [sqdist(samples.toarray()[i],cl.cluster_centers_[cl.labels_[i]]) for i in range(samples.shape[0])]
    c_sse = np.zeros(len(cl.cluster_centers_))
    c_n = np.zeros(len(cl.cluster_centers_)) #number of samples in each cluster
    cc = samples.toarray().mean(axis=0) #overall centroid
    ssb = 0

    print("Computing clustering quality SSE, SSB, STDERR ...")    

    for i in range(samples.shape[0]):
        c_n[cl.labels_[i]] += 1 
    for c in range(len(cl.cluster_centers_)): #for all clusters
        ssb += c_n[c]*sqdist(cl.cluster_centers_[c],cc) # compute total SSB
        for i in range(samples.shape[0]):
            if cl.labels_[i] == c:
                c_sse[c] += d[i]
                
    c_stderr = [(c_sse[c]**(1/2))/c_n[c] for c in range(n)] #standard error for each cluster (standard deviation)
#    print(cc,ssb)
    #TODO compute point wise and overall silhouette coeff
    #print("labels:",len(cl.labels_), max(cl.labels_))
    #print("samples:",samples.shape)
    print("Computing clustering quality Silhouette Factor ...")    
    s_sil = -1*np.ones(samples.shape[0]) #initialize to all negative (worst value)
    c_sil = -1*np.ones(n) #initialize to all negative (worst value)
    sil = -1
    if (False) and (samples.shape[0] > n) and (max(cl.labels_) >= 1) : #if samples more than clusters and at least two clusters 
        print("samples more than clusters and at least two clusters ...") 
        s_sil = silhouette_samples(samples.toarray(),cl.labels_)
        print("TOTAL SILHOUTTE = ",s_sil)
        print("Per Cluster Silhouette Factor ...")    
        for i in range(samples.shape[0]): #cluster silhouette
            c_sil[cl.labels_[i]] += s_sil[i]
        c_sil = [c_sil[i]/c_n[i] for i in range(n)]
        c_sil = np.array(c_sil)
        sil = silhouette_score(samples.toarray(),cl.labels_)
        print(sil)
    outparams = {'sample silhouette':s_sil,'cluster silhouette':c_sil,'total silhouette':sil,'total ssb':ssb,'cluster size':c_n,'centroids':cl.cluster_centers_, 'total sse':cl.inertia_, 'cluster sse':c_sse, 'squared distance':d, 'cluster stderr':c_stderr}
    #print(outparams)
    print("kmeans_clustering_end")
    return cl.labels_, outparams 
    
#TODO
def dbscan_clustering(samples):
#    cl = {}
#    return cl #dict {"traceName/VMID/CR3":[cluster_label, optional_param1, ...]}
    return

#TODO
def agglomerative_clustering(samples):
#    cl = {}
#    return cl #dict {"traceName/VMID/CR3":[cluster_label, optional_param1, ...]}
    return

#TODO
#def bipartite_community_clustering(samples):
#    cl = {}
#    return cl #dict {"traceName/VMID/CR3":[cluster_label, optional_param1, ...]}

#TODO
#def other_clustering():
#    cl = {}
#    return cl #dict {"traceName/VMID/CR3":[cluster_label, optional_param1, ...]}


class Clustering(Enum):
    KMEANS = 1
    AGGLOMERATIVE = 2
    DBSCAN = 3
    switcher = {
            KMEANS : kmeans_clustering,
            AGGLOMERATIVE : agglomerative_clustering,
            DBSCAN : dbscan_clustering
            }
            
 
class Vectorizer(Command):
    _DESC = """The vectorizer command."""
    _ANALYSIS_CLASS = vectorizer.Vectorizer
    _MI_TITLE = 'VM/PID Wait and Run Durations'
    _MI_DESCRIPTION = 'Per-VMID/PID, average and frequency of wait and run periods'
    _MI_TAGS = [mi.Tags.CPU, mi.Tags.TOP] #?
    _MI_TABLE_CLASS_PROC_FEATURE_VECTOR = 'VM/Process' #this goes in json class names for the output tables. These will be the top level tables 
    _MI_TABLE_CLASS_VCPU_FEATURE_VECTOR = 'VM/vCPU' #this goes in json class names for the output tables. These will be the top level tables 
    _MI_TABLE_CLASS_CLUSTERS = 'Clusterings' #this goes in json class names for the output tables. These will be the top level tables 
    _VECTOR_ORDER = ['TIMER','DISK','NET','TASK','OTHER','NON_ROOT','ROOT','L0_PREEMPTION']    
    _MI_TABLE_CLASSES = [
        (
            _MI_TABLE_CLASS_PROC_FEATURE_VECTOR,
            'Avg. Duration and Freq. of Wait/Run Periods', [#result table tab title in tracecompass (TC)
                #1st item:name of python variable holding the value for this column | 2nd:Column title | 3rd:Type of object which is the value of the column
                ('name', 'Experiment', mi.String),
                ('vmcr3', 'VMID/CR3', mi.String),
                ('timestamp', 'Timestamp', mi.String),
                ('avg_timer', 'Timer Wait', mi.Number),
                ('freq_timer', 'Timer Freq.', mi.Number),
                ('avg_disk', 'Disk Wait', mi.Number),
                ('freq_disk', 'Disk Freq.', mi.Number),
                ('avg_net', 'Net Wait', mi.Number),
                ('freq_net', 'Net Freq.', mi.Number),
                ('avg_task', 'Task Wait', mi.Number),
                ('freq_task', 'Task Freq.', mi.Number),
                ('avg_unknown', 'Unknown Wait', mi.Number),
                ('freq_unknown', 'Unknown Freq.', mi.Number),
                ('avg_nonroot', 'NonRoot Wait', mi.Number),
                ('freq_nonroot', 'NonRoot Freq.', mi.Number),
                ('avg_root', 'Root Wait', mi.Number),
                ('freq_root', 'Root Freq.', mi.Number),
                ('avg_l0', 'L0 Pre Wait', mi.Number),
                ('freq_l0', 'L0 Pre Freq.', mi.Number),
            ]
        ),
        (
            _MI_TABLE_CLASS_VCPU_FEATURE_VECTOR,
            'Avg. Duration and Freq. of Wait/Run Periods', [#result table tab title in tracecompass (TC)
                #1st item:name of python variable holding the value for this column | 2nd:Column title | 3rd:Type of object which is the value of the column
                ('name', 'Experiment', mi.String),
                ('vmvcpu', 'VMID/vCPU', mi.String),
                ('timestamp', 'Timestamp', mi.String),
                ('avg_timer', 'Timer Wait', mi.Number),
                ('freq_timer', 'Timer Freq.', mi.Number),
                ('avg_disk', 'Disk Wait', mi.Number),
                ('freq_disk', 'Disk Freq.', mi.Number),
                ('avg_net', 'Net Wait', mi.Number),
                ('freq_net', 'Net Freq.', mi.Number),
                ('avg_task', 'Task Wait', mi.Number),
                ('freq_task', 'Task Freq.', mi.Number),
                ('avg_unknown', 'Unknown Wait', mi.Number),
                ('freq_unknown', 'Unknown Freq.', mi.Number),
                ('avg_nonroot', 'NonRoot Wait', mi.Number),
                ('freq_nonroot', 'NonRoot Freq.', mi.Number),
                ('avg_root', 'Root Wait', mi.Number),
                ('freq_root', 'Root Freq.', mi.Number),
                ('avg_l0', 'L0 Pre Wait', mi.Number),
                ('freq_l0', 'L0 Pre Freq.', mi.Number),
                ('freq_vm_preempt', 'VM/VM Preemption Freq.', mi.Number),
                ('freq_host_preempt', 'VM/Host Preemption Freq.', mi.Number),
                ('freq_proc_preempt', 'In VM Proc Preemption Freq.', mi.Number),
                ('freq_thrd_preempt', 'In VM Thrd Preemption Freq.', mi.Number),
            ]
        ),
        (
            _MI_TABLE_CLASS_CLUSTERS,
            'Clusters', [#result table tab title in tracecompass (TC)
                #1st item:name of python variable holding the value for this column | 2nd:Column title | 3rd:Type of object which is the value of the column
                ('name', 'Experiment', mi.String),
                ('vmcr3', 'VMID/CR3', mi.String),
                ('km', 'KMEANS', mi.Number),
#                ('dbs', 'DBSCAN', mi.Number),
#                ('agg', 'AGGLOMERATIVE', mi.Number),
                #('avg_timer', 'Timer Wait', mi.Number),
                ('freq_timer', 'Timer Freq.', mi.Number),
                #('avg_disk', 'Disk Wait', mi.Number),
                ('freq_disk', 'Disk Freq.', mi.Number),
                #('avg_net', 'Net Wait', mi.Number),
                ('freq_net', 'Net Freq.', mi.Number),
                #('avg_task', 'Task Wait', mi.Number),
                ('freq_task', 'Task Freq.', mi.Number),
                #('avg_unknown', 'Unknown Wait', mi.Number),
                ('freq_unknown', 'Unknown Freq.', mi.Number),
                #('avg_nonroot', 'NonRoot Wait', mi.Number),
                ('freq_nonroot', 'NonRoot Freq.', mi.Number),
                #('avg_root', 'Root Wait', mi.Number),
                ('freq_root', 'Root Freq.', mi.Number),
                #('avg_idle', 'Idle Wait', mi.Number),
                ('freq_idle', 'Idle Freq.', mi.Number),
            ]
        ),
    ]

    #this is called when the analysis finishes
    #call heirarchy:
    #analysis.py: end_analysis()
    #analysis.py:   _send_notification_cb(AnalysisCallbackType.TICK_CB, None, end_ns=self._last_event_ts)
    #command.py:        _analysis_tick_cb(self, period, end_ns)
    #vectorizer.py:         _analysis_tick(self, period_data, end_ns)
    def _analysis_tick(self, period_data, end_ns): 
        if period_data is None:
            return
        
        #print("_analysis_tick\n")

        #determine period of analysis 
        begin_ns = period_data.period.begin_evt.timestamp 
        
        #populate list of clustering algorithms
        
        alg_list = create_alg_list(self._args)
#       alg_list = [ (Clustering.KMEANS,{'n_clusters':3,...},KMEANS_3),(),...]
            

        #register result tables in list of results 
        if self._mi_mode: #LAMI mode
            #with open('/home/azhari/FROM_UBUNTU/runtime-EclipseApplication/vm_analysis/.tracing/folder_list.txt') as listF:
            with open(self._args.list) as listF:
                traceNames = listF.readlines()
                traceNames = list(set(traceNames)) #remove duplicates
                traceNames = [fl.replace('\n','') for fl in traceNames] #remove newline at the end
                d_proc = {}
                avgvec_proc = {}
                fvec_proc = {}
                tracet_proc = {} #trace time
                prvec_proc = {} #Process preemption vector VM/VM, VM/Host, VM_internal_processes, VM_internal_threads
                exvec_proc = {} #Process exit vector 0:64 exit reasons
                d_vcpu = {}
                avgvec_vcpu = {}
                fvec_vcpu = {}
                tracet_vcpu = {} #trace time
                prvec_vcpu = {} #VM preemption vector VM/VM, VM/Host, VM_internal_processes, VM_internal_threads
                exvec_vcpu = {} #exit vector 0:64 exit reasons
                #process all experiments >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                for tr in traceNames:
                    path = self._args.list[0:[i for i in range(len(self._args.list)) if self._args.list[i]=='/'][-1]]
                    #print(path)
                    
                    trace_list = os.scandir(path+'/'+tr) #iterator for files in this folder
                    time_set = []
                    for f in trace_list:
                        if (f.name.split('[')[0] != 'net') and (f.name.split('[')[0] != 'disk'):
                            time_set.append(f.name.split('[')[1].split(']')[0]) #extract time string and update time_set (no repetitions!)
                    time_set = set(time_set)
                    #print(time_set)
                    
                    #iterate over time_set >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    for tt in time_set:
                        traceName = tr + ':' + tt
                        #print(traceName) #sva for test
                        procAvgFileName = path + '/' + tr + '/'+tr+'_requsts[' + tt + '].vector'
                        procFreqFileName = path + '/' + tr + '/'+tr+'_requsts[' + tt + '].vector'
                        procPreemptionFileName = procFreqFileName #path + '/' + tr + '/processInternal[' + tt + '].vector'
                        procExitFileName = procFreqFileName #path + '/' + tr + '/processExit[' + tt + '].vector'                    
                        vcpuAvgFileName = procAvgFileName #path + '/' + tr + '/cpuAvgdur[' + tt + '].vector'
                        vcpuFreqFileName = procFreqFileName #path + '/' + tr + '/cpuFrequency[' + tt + '].vector'
                        vcpuPreemptionFileName = procPreemptionFileName #path + '/' + tr + '/cpuInternal[' + tt + '].vector'
                        vcpuExitFileName = procExitFileName #path + '/' + tr + '/cpuExit[' + tt + '].vector'                    
                        
                        with open(procAvgFileName,'r') as procAvgF,\
                             open(procFreqFileName,'r') as procFreqF,\
                             open(vcpuAvgFileName,'r') as vcpuAvgF,\
                             open(vcpuFreqFileName,'r') as vcpuFreqF,\
                             open(vcpuPreemptionFileName,'r') as vcpuPreemptF,\
                             open(vcpuExitFileName,'r') as vcpuExitF,\
                             open(procPreemptionFileName,'r') as procPreemptF,\
                             open(procExitFileName,'r') as procExitF:
                            #returns dictionary with key = VMID/PID and values = wait times and frequencies all in one list
                            #TODO change vectorize_vcpu to a generic vectorize
                            d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc, tracet_proc = \
                                vectorize_vcpu(procAvgF,procFreqF,procPreemptF,procExitF,traceName,d_proc,avgvec_proc,fvec_proc,prvec_proc,exvec_proc,tracet_proc)
                                
                            d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu, tracet_vcpu = \
                                vectorize_vcpu(vcpuAvgF,vcpuFreqF,vcpuPreemptF,vcpuExitF,traceName,d_vcpu,avgvec_vcpu,fvec_vcpu,prvec_vcpu,exvec_vcpu,tracet_vcpu)
                            #no clustering just output feature vectors
                            ##feature_vector_table = \
                            ##    self._get_proc_feature_vector_result_table(period_data,begin_ns, end_ns, \
                            ##                                               traceName, d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc)
                            ##self._mi_append_result_table(feature_vector_table)
                            ##feature_vector_table = 
                            ##    self._get_vcpu_feature_vector_result_table(period_data,begin_ns, end_ns, 
                            ##                                               traceName, d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu)
                            ##self._mi_append_result_table(feature_vector_table)
                            ##names, clusters, samples, clustering_table, rvec_proc = \
                            ##    get_clusters(self, traceName, d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc,\
                            ##    d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu,\
                            ##    alg_list, self._args, begin_ns, end_ns)
                            ##if clusters != None:
                            ##    self._mi_append_result_table(clustering_table)
                    #end-for iterate over time_set <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                #end-for process all experiments <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                names, clusters, samples, clustering_table, rvec_proc = \
                    get_clusters(self, '', d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc, tracet_proc,\
                        d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu, tracet_vcpu,\
                        alg_list, self._args, begin_ns, end_ns)

                #get_random_data(samples.shape[0], samples.shape[1], alg_list, self._args)
                
                ##if clusters != None: 
                ##    self._mi_append_result_table(clustering_table)
                                 
                ##feature_vector_table = \
                ##        self._get_proc_feature_vector_result_table(period_data,begin_ns, end_ns, \
                ##                                                   '', d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc) #absolute frequency i.e., count
                ##self._mi_append_result_table(feature_vector_table)
                #feature_vector_table = 
                #        self._get_vcpu_feature_vector_result_table(period_data,begin_ns, end_ns, 
                #                                                   '', d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu)
                #self._mi_append_result_table(feature_vector_table)

                ##if self._args.rate == True:
                ##    feature_vector_table = \
                ##        self._get_proc_feature_vector_result_table(period_data,begin_ns, end_ns, '', \
                ##            d_proc, avgvec_proc, rvec_proc, prvec_proc, exvec_proc) #rate vector
                ##    self._mi_append_result_table(feature_vector_table)             
                       

        ##else: #non LAMI mode
        ##    self._print_feature_vector(feature_vector_table)

    #this is called after analysis is finished by the base class Command to create a summary table if required
    #to be overridden by subclasses 
    #implements code for populating optional result tables
    def _create_summary_result_tables(self): 
        self._mi_clear_result_tables() #no summary result table needed just clear all result tables 

    #now define all result table computing functions which were called in _analysis_tick() LAMI Mode
    #traceName is in traceName:timestamp format or equal to '' for aggregate result
    #TODO: add prvec and exvec contents
    def _get_proc_feature_vector_result_table(self, period_data, begin_ns, end_ns, traceName, d, avgvec, fvec, prvec, exvec):
        result_table = \
            self._mi_create_result_table(self._MI_TABLE_CLASS_PROC_FEATURE_VECTOR,
                                         begin_ns, end_ns)

        if traceName == '': #aggregate of all traces and times
            row_list = d.keys()
        else:#filter out this traceName:time and take related VM/CR3 values and put in list
            row_list = [s for s in d.keys() if s.split('/')[0] == traceName]
        
        for vmpid in row_list:#iterate over all VMID/PIDs
            tr_name = vmpid.split('/')[0].split(':')[0]
            timestamp = vmpid.split('/')[0].split(':')[1]
            vmid_cr3 = vmpid.split('/')[1]+'/'+vmpid.split('/')[2]
            result_table.append_row(
                name         = mi.String(tr_name),
                vmcr3        = mi.String(vmid_cr3),
                timestamp    = mi.String(timestamp),
                avg_timer    = mi.Number(avgvec[vmpid][0]),
                freq_timer   = mi.Number(fvec[vmpid][0]),
                avg_disk     = mi.Number(avgvec[vmpid][1]),
                freq_disk    = mi.Number(fvec[vmpid][1]),
                avg_net      = mi.Number(avgvec[vmpid][2]),
                freq_net     = mi.Number(fvec[vmpid][2]),
                avg_task     = mi.Number(avgvec[vmpid][3]),
                freq_task    = mi.Number(fvec[vmpid][3]),
                avg_unknown  = mi.Number(avgvec[vmpid][4]),
                freq_unknown = mi.Number(fvec[vmpid][4]),
                avg_nonroot  = mi.Number(avgvec[vmpid][5]),
                freq_nonroot = mi.Number(fvec[vmpid][5]),
                avg_root     = mi.Number(avgvec[vmpid][6]),
                freq_root    = mi.Number(fvec[vmpid][6]),
                avg_l0       = mi.Number(avgvec[vmpid][7]),
                freq_l0      = mi.Number(fvec[vmpid][7])
            )

        return result_table


    #traceName is in traceName:timestamp format or equal to '' for aggregate result
    #TODO Not implemented yet!
    def _get_vcpu_feature_vector_result_table(self, period_data, begin_ns, end_ns, traceName, d, avgvec, fvec, prvec, exvec):
        result_table = \
            self._mi_create_result_table(self._MI_TABLE_CLASS_PROC_FEATURE_VECTOR,
                                         begin_ns, end_ns)

        if traceName == '': #aggregate of all traces and times
            row_list = d.keys()
        else:#filter out this traceName:time and take related VM/CR3 values and put in list
            row_list = [s for s in d.keys() if s.split('/')[0] == traceName]
        
        for vmpid in row_list:#iterate over all VMID/PIDs
            tr_name = vmpid.split('/')[0].split(':')[0]
            timestamp = vmpid.split('/')[0].split(':')[1]
            vmid_cr3 = vmpid.split('/')[1]+'/'+vmpid.split('/')[2]
            result_table.append_row(
                name         = mi.String(tr_name),
                vmcr3        = mi.String(vmid_cr3),
                timestamp    = mi.String(timestamp),
                avg_timer    = mi.Number(avgvec[vmpid][0]),
                freq_timer   = mi.Number(fvec[vmpid][0]),
                avg_disk     = mi.Number(avgvec[vmpid][1]),
                freq_disk    = mi.Number(fvec[vmpid][1]),
                avg_net      = mi.Number(avgvec[vmpid][2]),
                freq_net     = mi.Number(fvec[vmpid][2]),
                avg_task     = mi.Number(avgvec[vmpid][3]),
                freq_task    = mi.Number(fvec[vmpid][3]),
                avg_unknown  = mi.Number(avgvec[vmpid][4]),
                freq_unknown = mi.Number(fvec[vmpid][4]),
                avg_nonroot  = mi.Number(avgvec[vmpid][5]),
                freq_nonroot = mi.Number(fvec[vmpid][5]),
                avg_root     = mi.Number(avgvec[vmpid][6]),
                freq_root    = mi.Number(fvec[vmpid][6]),
                avg_l0       = mi.Number(avgvec[vmpid][7]),
                freq_l0      = mi.Number(fvec[vmpid][7])
            )

        return result_table


    #add new command line arguments for this analysis
    def _add_arguments(self, ap):
        Command._add_vectorizer_args(ap)        
    
def get_random_data(n_samples, n_features, alg_list, args):
    print("get_random_data_begin", n_samples, n_features, alg_list)
    data = np.random.rand(n_samples, n_features)
    transformer = TfidfTransformer(norm=args.norm, smooth_idf=False, sublinear_tf=False, use_idf=False)
    data = transformer.fit_transform(data)
    cl = {} 
    trace_list = ['rnd{}'.format(i) for i in range(n_samples)] 
    
    sim = euclidean_sim(data)
    #data = data.toarray()

    for alg in alg_list: 
        func = Clustering.switcher.value.get(alg[0].value)
        c, param = func(data,alg[1]) #execute clustering algorithm over samples with alg[1] as inparams
        # c = list of cluster labels, where samples are ordered as in filtered_vmpid_list
        cl[ alg[2] ] = (c,param)

        #plot similarity matrix
        #TODO change according for new signature
        show_sim_matrix(sim,c,trace_list,'RAND_'+alg[2],param['total silhouette'])


#traceName is in this format: 'traceFileName:timestamp'
def vectorize_proc(avgF,freqF,traceName,d,avgvec,fvec):
    print('vectorize_proc_begin:',traceName)
    #avglines = avgF.readlines()
    freqlines = freqF.readlines()
    #avglines = [a.replace('\n','') for a in avglines]
    freqlines = [a.replace('\n','') for a in freqlines]
    #create a dictionary with key = 'traceFileName:timestamp/VMID/CR3' and values the wait times and frequencies all in one list
    tmpd={traceName+'/'+freqlines[i].split(',')[0] : \
        freqlines[i].split(',')[1:] for i in range(0,len(freqlines))}
    d.update(tmpd)
    
    #TODO I am making no use of the dict values, just the keys --> change into a list
    #TODO consider possibility of dict to index various fields, e.g., f['timer']=int(tmpd[vmpid][8])          
    for vmpid in tmpd.keys():
        f = np.zeros(9)
        f[0] = int(tmpd[vmpid][0]) #timer freq
        f[1] = int(tmpd[vmpid][1])
        f[2] = int(tmpd[vmpid][2])
        f[3] = int(tmpd[vmpid][3])
        f[4] = int(tmpd[vmpid][4])
        f[5] = int(tmpd[vmpid][5])
        f[6] = int(tmpd[vmpid][6])
        f[7] = int(tmpd[vmpid][7]) #L0 preemption freq
        f[8] = int(tmpd[vmpid][8]) #L0 preemption freq
        fvec[vmpid] = f
    print('vectorize_proc_end:',traceName)          
    return d, fvec  


#traceName is in this format: 'traceFileName:timestamp'
#TODO get total trace time from first line of Avgdur.vector
#TODO use exectime of entry as new feature
def vectorize_vcpu(avgF,freqF,preemptF,exitF,traceName,d,avgvec,fvec,prvec,exvec,traceTime):
    print('vectorize_vcpu_begin:',traceName)
     
    #avglines = avgF.readlines()
    freqlines = freqF.readlines()
    #prlines = preemptF.readlines()
    #exlines = exitF.readlines()
    #avglines = [a.replace('\n','') for a in avglines]
    freqlines = [a.replace('\n','') for a in freqlines]
    #traceT = int(freqlines[8]) #get total trace time from first line of Avgdur.vector
    #avglines = avglines[1:] #save the rest as average duration data
       
    #if len(prlines) < len(avglines):
    #    print("WARNING IN DATA: mismatch in number of entries",traceName)

    #if len(prlines) == 0: # fix error in input .vector file when it is empty
    #    prlines = [a.split(',')[0]+',0,0,0,0' for a in avglines]
    #    print("Mismatch automatically fixed by adding all zero entries")

    #prlines = [a.replace('\n','') for a in prlines]
    #exlines = [a.replace('\n','') for a in exlines]

    #create a dictionary with key = traceFileName:timestamp/VMID/VCPUID(or CR3 in case of proc) 
    #and values an array of (65) zeros to be later filled with exit counts
    #extmp={traceName+'/'+avglines[i].split(',')[0] : \
    #    np.zeros(65) for i in range(0,len(freqlines))}
    #exvec.update(extmp)
    #vectorize exit counts
    #for l in exlines: # l = VMID/vCPU(or CR3)/ExitNo,Count
    #    vmvcpu = traceName + '/' + l[0:l.rfind('/')] # = VMID/vCPU (or CR3)
    #    exno = int(l[l.rfind('/')+1:].split(',')[0]) # = ExitNo
    #    cnt = int(l[l.rfind('/'):].split(',')[1]) # = Count
    #    exvec[vmvcpu][exno] = cnt
        
    #create a dictionary with key = traceFileName:timestamp/VMID/VCPUID( or CR3) and values the wait times and frequencies all in one list
    #FIXME assumes same order in fre/avg/preemption files (fragile but works)
    tmpd={traceName+'/'+freqlines[i].split(',')[0] : \
        freqlines[i].split(',')[1:] \
        for i in range(0,len(freqlines))}
    
    #tmppr={ traceName+'/'+prlines[i].split(',')[0] : prlines[i].split(',')[1:] for i in range(len(prlines)) }
    #print(tmppr)

    #for key in tmpd.keys():
    #    if key in tmppr.keys():
    #        tmpd[key] = tmpd[key] + tmppr[key]
    #    else:
    #        tmpd[key] = tmpd[key] + ['0', '0', '0', '0', '0', '0', '0', '0'] #insert all zero data in place of missing entries
    #print(tmppr, tmpd)
    
    #print(tmpd)                   
    d.update(tmpd)

    #TODO consider possibility of dict to index various fields, e.g., f['timer']=int(tmpd[vmpid][8])          
    for vmvcpu in tmpd.keys():
        pr = np.zeros(8)
        f = np.zeros(9)
        avg = np.zeros(8)

        #avg[0] = int(tmpd[vmvcpu][0]) #timer avg
        #avg[1] = int(tmpd[vmvcpu][1]) 
        #avg[2] = int(tmpd[vmvcpu][2]) 
        #avg[3] = int(tmpd[vmvcpu][3]) 
        #avg[4] = int(tmpd[vmvcpu][4]) 
        #avg[5] = int(tmpd[vmvcpu][5]) 
        #avg[6] = int(tmpd[vmvcpu][6]) 
        #avg[7] = int(tmpd[vmvcpu][7]) #L0 preemption avg 

        f[0] = int(tmpd[vmvcpu][0]) #timer freq
        f[1] = int(tmpd[vmvcpu][1])
        f[2] = int(tmpd[vmvcpu][2])
        f[3] = int(tmpd[vmvcpu][3])
        f[4] = int(tmpd[vmvcpu][4])
        f[5] = int(tmpd[vmvcpu][5])
        f[6] = int(tmpd[vmvcpu][6])
        f[7] = int(tmpd[vmvcpu][7]) #L0 preemption freq
        f[8] = int(tmpd[vmvcpu][8]) #L0 preemption freq
        traceT = int(tmpd[vmvcpu][8])
        #pr[0] = int(tmpd[vmvcpu][16]) #VM/VM preemption freq
        #pr[1] = int(tmpd[vmvcpu][17]) #VM/Host preemption freq
        #pr[2] = int(tmpd[vmvcpu][18]) #In VM process preemption freq
        #pr[3] = int(tmpd[vmvcpu][19]) #In VM thread preemption freq
        #pr[4] = int(tmpd[vmvcpu][20]) #Inj_timer: when a timer interrupt is injected to process/VM
        #pr[5] = int(tmpd[vmvcpu][21]) #Inj_task: when a task interrupt is injected to process/VM
        #pr[6] = int(tmpd[vmvcpu][22]) #Inj_disk: when a disk interrupt is injected to process/VM
        #pr[7] = int(tmpd[vmvcpu][23]) #Inj_net: when a net interrupt is injected to process/VM
        
        traceTime[vmvcpu] = np.array([traceT]) #store as an array of size 1 to be consistent with other vectors
        #avgvec[vmvcpu] = avg
        fvec[vmvcpu] = f
        #prvec[vmvcpu] = pr
    
    #print('Freq Vector:',fvec,' duration= ',traceT)    
    #print('Freq Vector:',fvec)    
    #print('Preemption Vector:',prvec)    
        
          
    print('vectorize_vcpu_end:',traceName)
    return d, avgvec, fvec, prvec, exvec, traceTime 

# (weighted) sum of all rows belonging to a VM and returns the resulting vector
# input parameters:
# |vec|: input vector to be collapsed 
# |weight_vec|: optional weight vector. If none provided then simple sum.
# returns:
# |new_vec|: collapsed vector 
def collapse_vm_samples(vec, weight_vec = None):
    print("collapse_vm_samples")
    new_vec = {}
    total_weight = {} 
    for key in vec.keys():
        vm = key[0:key.rfind('/')]
        if vm in new_vec:
            if weight_vec == None:
                new_vec[vm] = new_vec[vm] + vec[key]
            else:
                new_vec[vm] = new_vec[vm] + vec[key]*weight_vec[key]
                total_weight[vm] = total_weight[vm] + weight_vec[key]            
        else:
            if weight_vec == None:            
                new_vec[vm] = vec[key]
            else:
                new_vec[vm] = vec[key]*weight_vec[key]
                total_weight[vm] = weight_vec[key]            
    
    if weight_vec != None:            
        for key in new_vec.keys():
            new_vec[key] = new_vec[key] / total_weight[key]
            
    return new_vec

# Take dictionary of vectors |vec| and list of feature indices to be considered |index_list|
# then select those vectors having one of the selected features as among the |top_n| values
# input parameters:
# |vec|: dictionary of all vectors (samples)
# |index_list|: list of designated feature indices
# |top_n|: samples having at least one designated feature among the top_n, to be selected as output
# |key_list|: list of keys for not yet selected samples in the dictionary 
# |sz|: size of each sample vector, i.e., max number of features
# returns:
# |fl|: newly filtered (selected) sample keys from the dict
# |fl_out|: filtered out (not selected) sample keys  
def filter_samples(top_n, index_list, key_list, vec, sz):
    #print("filter_samples_begin",key_list)#,sz, top_n, index_list, key_list, vec)
    n = top_n
    if n > len(key_list):
        n = len(key_list)
    
    #take samples among the top n in any of the features, 
    #n=0 means take all samples (TODO not tested): n<0 means the bottom samples 
    #TODO Another alternative is to take the top n AFTER normalization
    top = [sorted(np.array(list(vec.values()))[:,ii])[-n] for ii in range(sz)] #take top n freq feature values and store in topf
    #print(top)
    fl = []
    fl_out = list(key_list)
    #print(key_list)
    if len(index_list) > 0:
        for key in key_list:
            tmp = [ True for j in index_list if vec[key][j] >= top[j] ]
            #print(key,tmp)
            if any(tmp): #at least one column in vector satisfies filter criteria
                fl.append(key)
                fl_out.remove(key)

    #print("filter_samples_end",fl,fl_out)
    return fl, fl_out


# input parameters:
# |key_list|: list of keys for selected samples from the dictionary of vectors
# |vec_tuple|: a tuple holding sample vectors of interest
# |index_tuple|: tuple of list of designated feature indices
# returns:
# |sample_tuple|: a tuple with same size and order as |vec_tuple| containing 
#            numpy.array holding samples indicated in |key_list| and reduced by 
#            designated indices
def create_vectors(key_list, vec_tuple, index_tuple, norm_type):
    print("create_vectors_begin") #,key_list, vec_tuple, index_tuple)
    k = 0    
    it = index_tuple 
    samples = np.zeros( (len(key_list), len(it)) ) #initialize samples with zero vectors
    if len(it) > 0:
        i = 0 #index of rows (samples)
        vec = vec_tuple#[k]
        for key in key_list:
            samples[i,:] = [ vec[key][j] for j in it ]
            i += 1
        #perform feature vector normalization
        #TODO add option for no normalization
        if samples.shape[1] == 1: #no normalization if only one feature available FIXME could cause problems in logic
            norm_type = None
        if i > 0: #at least one sample remains after filtering
            transformer = TfidfTransformer(norm=norm_type, smooth_idf=False, sublinear_tf=False, use_idf=False)
            samples = transformer.fit_transform(samples)
            print("^^^^^^^^^^^^^")
            samples = samples.toarray()
    if k == 0:
        sample_tuple = (samples,)              
    else:
        sample_tuple = sample_tuple.__add__( (samples,) )    
    k = k+1
    #end of for it ...
    print("create_vectors_end")#,key_list, sample_tuple)
    return sample_tuple

# form aggregate sample matrix and re-normalize
# input parameters:
# |sample_tuple|: tuple including final sample matrices
# returns:
# |samples|: a unified and re-normalized sample numpy sparse array including all in the tuple
# do not call samples.toarray() in this function!
def create_sample_matrix(sample_tuple, norm_type):
    #print("create_sample_matrix_begin", sample_tuple)
    i = 0
    for it in sample_tuple:
        if len(it) != 0:
            if i == 0:
                samples = it;
            else:
                samples = np.append(samples,it,1)
            i = i+1
    
    #TODO add option for no normalization
    transformer = TfidfTransformer(norm=norm_type, smooth_idf=False, sublinear_tf=False, use_idf=False)
    samples = transformer.fit_transform(samples)
    #print("create_sample_matrix_end", samples.toarray())
    return samples


#TODO add input params for prvec_proc and exvec_proc
def get_clusters(vectorizer, traceName, d_proc, avgvec_proc, fvec_proc, prvec_proc, exvec_proc, tracet_proc,\
                d_vcpu, avgvec_vcpu, fvec_vcpu, prvec_vcpu, exvec_vcpu, tracet_vcpu, alg_list, args, begin_ns, end_ns):
    print("get_clusters_begin",traceName)
    if traceName == '': #aggregate of all traces
        vmpid_list = d_proc.keys() #needs to be overwritten after collapse_vm_samples
        #vmvcpu_list = d_vcpu.keys() #needs to be overwritten after collapse_vm_samples
    else:#filter out this traceName and take related VM/CR3 values and put in list
        vmpid_list = [s for s in d_proc.keys() if s.split('/')[0] == traceName] #needs to be overwritten after collapse_vm_samples
        #vmvcpu_list = [s for s in d_vcpu.keys() if s.split('/')[0] == traceName] #needs to be overwritten after collapse_vm_samples
    
    #build sample matrix out of feature vectors begin >>>>>>>>>>>>>>>>>>>>>>>>>
    #preprocess feature vectors (filtering)
    #determine which features to consider (Freq|Wait)
    #(TIMER,DISK,NET,TASK,OTHER,NON_ROOT,ROOT,IDLE,L0_PREEMPTION)
    f_proc_index = [] #will eventually contain index number for feature to be included in analysis
    #w_proc_index = []
    #exec_proc_index = []
    #pr_proc_index = [] #will eventually contain index number for feature to be included in analysis
    #ex_proc_index = [] #will eventually contain index number for feature to be included in analysis

    #f_vcpu_index = [] #will eventually contain index number for feature to be included in analysis
    #w_vcpu_index = []
    #exec_vcpu_index = []
    #pr_vcpu_index = [] #will eventually contain index number for feature to be included in analysis
    #ex_vcpu_index = [] #will eventually contain index number for feature to be included in analysis
    

    #dict of command line arguments and the corresponding feature index to take
    #(w)ait and (f)requency vector: (ti)mer/(ta)sk/...  
    index = {'fti':0,'wti':0,'fdi':1,'wdi':1,'fne':2,'wne':2,'fta':3,'wta':3,'fot':4,'wot':4,'fno':5,'wno':5,'fro':6,'wro':6,'fl0':7,'wl0':7}
    #preemption vector: VM/VM, VM/Host, In VM Process, In VM Thread, (i)nject (ti)mer/(ta)sk/(di)sk/(ne)twork
    #prindex = {'pvm':0,'pho':1,'ppr':2,'pth':3, 'piti':4, 'pita':5, 'pidi':6, 'pine':7}
    fw_names = {0:'timer',1:'disk',2:'net',3:'task',4:'other',5:'non-root',6:'root',7:'l0'} 
    #pr_names = {0:'vm',1:'host',2:'proc',3:'thread',4:'timer',5:'task',6:'disk',7:'net'} 

    #--proc: process based feature selection
    for feature in args.proc.split(','): 
        if feature == '':
            pass
        elif feature[0] == '*':
            f_proc_index = list(range(max(index.values())+1)) #max number of features obtained from index dict
            #w_proc_index = list(range(max(index.values())+1))
            #exec_proc_index = [0] 
            #pr_proc_index = list(range(max(prindex.values())+1))
            #ex_proc_index = list(range(65))
        
        elif feature[1] == '*':      
            if feature[0] == 'f':
                f_proc_index = list(range(max(index.values())+1))
            #if feature[0] == 'w':
            #    w_proc_index[:] = list(range(max(index.values())+1))
            #if feature[0] == 'p':
            #    pr_proc_index[:] = list(range(max(prindex.values())+1))
            #if feature[0] == 'e':
            #    ex_proc_index[:] = list(range(65))
        else:
            if feature[0] == 'f':
                f_proc_index.append(index[feature])
            #if feature[0] == 'w':
            #    w_proc_index.append(index[feature])
            #if feature[0] == 'xe': #execution time (just xe)
            #    exec_proc_index = [0]
            #if feature[0] == 'p':
            #    pr_proc_index.append(prindex[feature])
            #if feature[0] == 'e':
            #    for ex_reason in feature[1:].split('.'):
            #        ex_proc_index.append(int(ex_reason))
    
                    
                    
    print("Selected indices:")
    print("Proc:",f_proc_index)#, w_proc_index, pr_proc_index, ex_proc_index, exec_proc_index)
    #print("vCPU:",f_vcpu_index, w_vcpu_index, pr_vcpu_index, ex_vcpu_index, exec_vcpu_index)

    rvec_proc = {}
    execvec_proc = {} #dict holding total execution time per VM/process in nsec
    #rvec_pr_proc = {}
    #rvec_ex_proc = {}
    #rvec_exec_proc = {}

    #rvec_vcpu = {}
    #execvec_vcpu = {} #dict holding total execution time per VM/vCPU in nsec
    #rvec_pr_vcpu = {}
    #rvec_ex_vcpu = {}
    #rvec_exec_vcpu = {}
    #TODO refactor into one function call per proc and vcpu
    for vmpid in vmpid_list:
        exec_time = fvec_proc[vmpid][8]
        execvec_proc[vmpid] = exec_time
        #rvec_exec_proc[vmpid] = execvec_proc[vmpid] / tracet_proc[vmpid] #execution time as fraction of trace time
        if exec_time == 0:
            print('WARNING: Zero execution time in: ', vmpid)
            if fvec_proc[vmpid][0:7].any():
                print('WARNING: Bad frequency vector in: ', vmpid)
            else:
                rvec_proc[vmpid] = fvec_proc[vmpid] 
                #rvec_pr_proc[vmpid] = prvec_proc[vmpid]
                #rvec_ex_proc[vmpid] = exvec_proc[vmpid] 
        else:
            rvec_proc[vmpid] = 1000000000 * (fvec_proc[vmpid] / exec_time) #per nanosec to per sec
            #rvec_pr_proc[vmpid] = 1000000000 * (prvec_proc[vmpid] / exec_time) #per nanosec to per sec
            #rvec_ex_proc[vmpid] = 1000000000 * (exvec_proc[vmpid] / exec_time) #per nanosec to per sec
     
    print("===========")
    #if --rate is provided then obtain waiting rate instead of waiting frequency
    #waiting frequency is absolute total number of times the entity has waited during trace period    
    #waiting rate is number of times the entity has waited per second
    #only apply to frequency features
    #TODO: to be applied separately to preemption/exit/freq features using appropriate cmd args
    #TODO bad coding, overwritting previously computed rvec
    if args.rate == False:
        for vmpid in vmpid_list:
            rvec_proc[vmpid] = fvec_proc[vmpid]
            #rvec_pr_proc[vmpid] = prvec_proc[vmpid]
            #rvec_ex_proc[vmpid] = exvec_proc[vmpid]
            #rvec_exec_proc[vmpid] = execvec_proc[vmpid]

    print("........")
    filtered_vmpid_list, fl_out = filter_samples(args.top, f_proc_index, list(vmpid_list), rvec_proc, max(index.values())+1 )
    print("........")
    
    #fl, fl_out = filter_samples(args.top, w_proc_index, fl_out, avgvec_proc, max(index.values())+1 )
    #filtered_vmpid_list = filtered_vmpid_list + fl
    #fl, fl_out = filter_samples(args.top, pr_proc_index, fl_out, rvec_pr_proc, max(prindex.values())+1 )
    #filtered_vmpid_list = filtered_vmpid_list + fl
    ##print(filtered_vmpid_list, fl_out)
    #fl, fl_out = filter_samples(args.top, ex_proc_index, fl_out, rvec_ex_proc, 65)
    #filtered_vmpid_list = filtered_vmpid_list + fl
    #fl, fl_out = filter_samples(args.top, exec_proc_index, fl_out, rvec_exec_proc, 1)
    #filtered_vmpid_list = filtered_vmpid_list + fl
    ##print(filtered_vmpid_list, fl_out)
    ##print("*********")
    (f_proc_samples) = \
        create_vectors(\
            filtered_vmpid_list, \
            (rvec_proc), \
            (f_proc_index), \
            args.norm \
            )
    
    print("*********")
    
    if len(filtered_vmpid_list) == 0: #no samples made it through
        return None, None, None, None
    
    if len(filtered_vmpid_list) != 0:
        samples_proc = create_sample_matrix( (f_proc_samples), args.norm )
        #now compute similarity matrix among samples
        #TODO run on a separate thread and join
        sim_proc = euclidean_sim(samples_proc)

    #build sample matrix out of feature vectors end <<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    #compute clustering and create result table begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    cl_proc = {} #for processXXXX.vector files
    if len(filtered_vmpid_list) != 0:
        for alg in alg_list: 
            func = Clustering.switcher.value.get(alg[0].value)
            c, param = func(samples_proc,alg[1]) #execute clustering algorithm over samples with alg[1] as inparams
            # c = list of cluster labels, where samples are ordered as in filtered_vmpid_list
            cl_proc[ alg[2] ] = (c,param)

            #plot similarity matrix
            #TODO in multithreaded version showing similarity matrix can be performed on a different thread along with sim mat computation after computing all clusters
            if traceName == '':
                show_sim_matrix(sim_proc, filtered_vmpid_list, 'Proc_'+alg[2], samples_proc, cl_proc[ alg[2] ])

    cl = cl_proc
    filtered_vm_list= []
    if (traceName == '') and (len(filtered_vm_list) != 0): #VM analysis only makes sense for aggregate of trace files
        cl_vm = {} #for cpuXXXX files, where we have already collapsed all vCPUs to get a VM view
        for alg in alg_list: 
            func = Clustering.switcher.value.get(alg[0].value)
            if ('KMEANS_3' in cl_vm.keys()):
                print("==================================")
                labels_km3 = (cl_vm[ 'KMEANS_3' ][0]).tolist()
                sim_vm_list = sim_vm.tolist()
                print("*************************************")
                for ii in range(3):
                    alg2 = alg[2]+'_C{}'.format(ii)
                    kk = 0
                    samples_vm_flt = np.zeros((0,samples_vm.shape[1]))
                    filtered_vm_list_new = []
                    sim_vm_flt = []
                    indx = []
                    for jj in range(len(labels_km3)):
                        if labels_km3[jj] == ii:
                            #print("111111111111")#, samples_vm_flt, np.array([ samples_vm.toarray()[jj] ]) )
                            samples_vm_flt = np.append(samples_vm_flt, np.array([ samples_vm.toarray()[jj] ]), 0)
                            filtered_vm_list_new.append(filtered_vm_list[jj])
                            indx.append(jj)
                            kk += 1
                    print(ii,":",indx)
                    for rr in range(kk):
                        sim_vm_flt.append([sim_vm_list[indx[rr]][indx[pp]] for pp in range(kk)])
                        #print(sim_vm_flt[rr])
                    samples_vm_flt = sparse.csr_matrix(samples_vm_flt)
                    c, param = func(samples_vm_flt, alg[1])
                    cl_vm[ alg2 ] = (c,param)
                    #plot similarity matrix
                    if traceName == '':
                        show_sim_matrix(np.array(sim_vm_flt),filtered_vm_list_new,'VM_'+alg2, samples_vm_flt, cl_vm[ alg2 ], legend_vm, short_name = True)
            else:
                alg2 = alg[2]    
                c, param = func(samples_vm,alg[1]) #execute clustering algorithm over samples with alg[1] as inparams
                cl_vm[ alg2 ] = (c,param)  
                #plot similarity matrix
                if traceName == '':
                    show_sim_matrix(sim_vm,filtered_vm_list,'VM_'+alg2, samples_vm, cl_vm[ alg2 ], legend_vm, short_name = True)
#        if ('KMEANS_3' in cl_vm.keys()) and ('KMEANS_15' in cl_vm.keys()):
#            print("Combination analysis KMEANS 3 and 15")
#            show_sim_matrix_filter(sim_vm,filtered_vm_list,'VM_'+alg[2], samples_vm, cl_vm[ 'KMEANS_15' ], cl_vm[ 'KMEANS_3' ], legend_vm)

    #overall best VM and Proc clustering results
    if (traceName == '') and (len(filtered_vm_list) != 0) and (len(filtered_vmpid_list) != 0):         
        max_sil_vm = 0
        for key in cl_vm.keys():
            if max_sil_vm < cl_vm[key][1]['total silhouette']:
                max_sil_vm = cl_vm[key][1]['total silhouette']
                max_sil_vm_key = key
        
        max_sil_proc = 0
        for key in cl_proc.keys():
            if max_sil_proc < cl_proc[key][1]['total silhouette']:
                max_sil_proc = cl_proc[key][1]['total silhouette']
                max_sil_proc_key = key
        
        jj = 0
        proc_vm_label = {}
        #print(filtered_vm_list)
        #print(filtered_vmpid_list)
        for l in cl_proc[max_sil_proc_key][0]: #for l in cluster labels of selected proc clustering
            vm = filtered_vmpid_list[jj][0:filtered_vmpid_list[jj].rfind('/')] #get the vm key: traceName:timestamp/vmpid
            #print(vm, filtered_vm_list.index(vm), cl_vm[max_sil_vm_key][0])
            proc_vm_label[vm] = cl_vm[max_sil_vm_key][0][filtered_vm_list.index(vm)]
            jj = jj + 1
        
        print(proc_vm_label)
        ##TODO fix so that uses sim_proc,sim_vm as input: show_sim_matrix_proc_vm(proc_vm_label, samples_vm, cl_vm[max_sil_vm_key], filtered_vm_list, samples_proc, cl_proc[max_sil_proc_key], filtered_vmpid_list)

        #show clustering results for procXXXX.vector files
#        show_clustering_results(title = 'VM Clustering',\
#            col_infos = col_infos_vm,\
#            clusters = cl_vm,\
#            samples = samples_vm,\
#            algs = alg_list,\
#            feature_index = (f_vcpu_index, w_vcpu_index, pr_vcpu_index, ex_vcpu_index),\
#            rows = filtered_vm_list,\
#            begin_ts = begin_ns, end_ts = end_ns\
#            )

    #TODO: Refactor
    #show clustering results for procXXXX.vector files
#    show_clustering_results(title = 'Proc Clustering',\
#        col_infos = col_infos_proc,\
#        clusters = cl_proc,\
#        samples = samples_proc,\
#        algs = alg_list,\
#        feature_index = (f_proc_index, w_proc_index),\
#        rows = filtered_vmpid_list,\
#        begin_ts = begin_ns, end_ts = end_ns\
#        )

    
    #construct table structure and columns
    col_infos = [
            ('name', 'Experiment', mi.String),
            ('vmcr3', 'VMID/CR3', mi.String),            
            ('timestamp', 'Time', mi.String),
            ]
    
    col_infos_vm = [
            ('name', 'Experiment', mi.String),
            ('vmpid', 'VMID', mi.String),            
            ('timestamp', 'Time', mi.String),
            ]
    
    #compute clusterings 
    #populate cl = {'KMEANS':(clusterlabels[1,2,3,1], paramlist[]), ...}
    #iterate over list of clustering algorithms
    i = 0;
    for alg in alg_list: 

        col_infos.append((
            'alg{}'.format(i),
            alg[2],
            mi.Number
        ))
        col_infos.append((
            'd{}'.format(i),
            'SqDist',
            mi.Number
        ))
        col_infos.append((
            'sse{}'.format(i),
            'CL SSE',
            mi.Number
        ))
        col_infos.append((
            'sil{}'.format(i),
            'Silhouette',
            mi.Number
        ))
        i += 1
        
    i=0
    for f in f_proc_index:
        col_infos.append((
            'c{}'.format(i),
            vectorizer._VECTOR_ORDER[f]+' Freq.',
            mi.Number
        ))
        i += 1
    for w in f_proc_index:
        col_infos.append((
            'c{}'.format(i),
            vectorizer._VECTOR_ORDER[w]+' Wait',
            mi.Number
        ))
        i += 1
    title = 'Proc Clustering'
    table_class = mi.TableClass(None, title, col_infos)
    result_table = mi.ResultTable(table_class, begin_ns, end_ns)

#    title_vcpu = 'VM Clustering'
#    table_class_vcpu = mi.TableClass(None, title_vcpu, col_infos_vcpu)
#    result_table_vcpu = mi.ResultTable(table_class_vcpu, begin_ns, end_ns)

    print("#populate rows")
    return filtered_vmpid_list, cl, samples_proc, result_table, rvec_proc
    i=0
    samples_arr = samples_proc.toarray() #change from sparse matrix to numpy array so can be indexed properly       
    for vmpid in filtered_vmpid_list:#iterate over all VMID/PIDs
        tr_name = vmpid.split('/')[0]
        vmid_cr3 = vmpid.split('/')[1]+'/'+vmpid.split('/')[2]
        row_tuple = [
            mi.String(tr_name),
            mi.String(vmid_cr3),
        ]
        
        for alg in alg_list:
            label = int(cl[ alg[2] ][0][i])
            row_tuple.append( mi.Number( label ) ) #add cluster label
            dist = cl[ alg[2] ][1]['squared distance']
            row_tuple.append( mi.Number( dist[i] ) ) #add sample distance to cluster center
            c_sse = cl[ alg[2] ][1]['cluster sse']
            row_tuple.append( mi.Number( c_sse[label] ) ) #add cluster SSE for this sample's cluster
            s_sil = cl[ alg[2] ][1]['sample silhouette'][i]
            row_tuple.append( mi.Number( s_sil ) ) #add sample silhouette for this sample
                    
        for col in range(len(f_proc_index)+len(f_proc_index)):
            row_tuple.append(mi.Number(samples_arr[i][col]))
        result_table.append_row_tuple(tuple(row_tuple))
        i += 1   

    print("#add row containing clustering SSE ")
    row_tuple = [
        mi.String('Total SSE'),
        mi.String('N/A'),
    ]
    for alg in alg_list:
        row_tuple.append( mi.Number( float(cl[ alg[2] ][1]['total sse']) ) ) #[1:outparams][1:inertia/SSE] take inertia among all samples
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )

    for col in range(len(f_proc_index)+len(f_proc_index)):
        row_tuple.append(mi.Number(0))
    result_table.append_row_tuple(tuple(row_tuple))
    
    print("#add row containing clustering total SSB")
    row_tuple = [
        mi.String('Total SSB'),
        mi.String('N/A'),
    ]
    for alg in alg_list:
        row_tuple.append( mi.Number( float(cl[ alg[2] ][1]['total ssb']) ) ) #[1:outparams][1:inertia/SSE] take inertia among all samples
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )

    for col in range(len(f_proc_index)+len(f_proc_index)):
        row_tuple.append(mi.Number(0))
    result_table.append_row_tuple(tuple(row_tuple))
    
    print("#add row containing clustering total silhouette")
    row_tuple = [
        mi.String('Total Silhouette'),
        mi.String('N/A'),
    ]
    for alg in alg_list:
        row_tuple.append( mi.Number( float(cl[ alg[2] ][1]['total silhouette']) ) ) #[1:outparams][1:inertia/SSE] take inertia among all samples
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )
        row_tuple.append( mi.Number( 0 ) )

    for col in range(len(f_proc_index)+len(f_proc_index)):
        row_tuple.append(mi.Number(0))
    print("&&&&&&&&&&&&&")    
    result_table.append_row_tuple(tuple(row_tuple))
    #TODO: end refactor
    
    #compute clustering and create result table end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print("get_clusters_end",traceName)
    return filtered_vmpid_list, cl, samples_proc, result_table, rvec_proc


#creates custom names for various clustering algorithms based on their input parameters
#e.g., (Clustering.KMEANS, {'n_clusters':3}, KMEANS_3)
#TODO to be modified for each new algorithm
def create_alg_list(args):
    tmp = []
    for arg in args.algs.split(','):
        if arg[0:6] == 'kmeans':
            alg = Clustering.KMEANS
            params = {'n_clusters':int(arg[6:])} #number of clusters
            col_name = alg.name + '_' + arg[6:]
#        elif args == 'dbscan':
#        elif args == 'aggmin':
#        elif args == 'aggavg':
#        elif args == 'aggmax':
#        else:
        tmp.append( (alg,params,col_name) )    

    return tmp

def _run(mi_mode):
    vectorizercmd = Vectorizer(mi_mode=mi_mode)
    vectorizercmd.run() #this triggers the whole analysis part 
    #TODO to bypass the analysis we will call _analysis_tick(self, period_data, end_ns)
    #TODO instead. But first we need to create its input arguments: period_data, end_ns
    #vectorizercmd._analysis_tick(period_data = None, end_ns = 0)
    #vectorizercmd._post_analysis()
    


def run():
    _run(mi_mode=False)


def run_mi():
    _run(mi_mode=True)
