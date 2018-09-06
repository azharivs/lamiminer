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

import operator
from ..common import format_utils
from .command import Command
from ..core import vectorizer
from . import mi
from . import termgraph

from sklearn.cluster import KMeans
from enum import Enum
import numpy as np

#performs kmeans clustering on a list of samples
#TODO set number of clusters via parameter
def kmeans_clustering(samples):
    n = min (10, samples.shape[0]) #number of clusters never larger than number of samples
    cl = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1, random_state=None)
    cl.fit(samples)
    outparams = []
    #print(cl)
    #print(samples)
    #print(cl.labels_)
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
    _MI_TABLE_CLASS_FEATURE_VECTOR = 'Wait Period Vectorization' #this goes in json class names for the output tables. These will be the top level tables 
    _MI_TABLE_CLASS_CLUSTERS = 'Clusterings' #this goes in json class names for the output tables. These will be the top level tables 
    _MI_TABLE_CLASSES = [
        (
            _MI_TABLE_CLASS_FEATURE_VECTOR,
            'Avg. Duration and Freq. of Wait/Run Periods', [#result table tab title in tracecompass (TC)
                #1st item:name of python variable holding the value for this column | 2nd:Column title | 3rd:Type of object which is the value of the column
                #TODO add a VM class as well
                #('path','Path', mi.String),
                ('name', 'Experiment', mi.String),
                ('vmcr3', 'VMID/CR3', mi.String),
 #               ('cluster', 'Cluster', mi.Number),
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
                ('avg_idle', 'Idle Wait', mi.Number),
                ('freq_idle', 'Idle Freq.', mi.Number),
            ]
        ),
        (
            _MI_TABLE_CLASS_CLUSTERS,
            'Clusters', [#result table tab title in tracecompass (TC)
                #1st item:name of python variable holding the value for this column | 2nd:Column title | 3rd:Type of object which is the value of the column
                #TODO add a VM class as well
                #('path','Path', mi.String),
                ('name', 'Experiment', mi.String),
                ('vmcr3', 'VMID/CR3', mi.String),
                ('km', 'KMEANS', mi.Number),
#                ('dbs', 'DBSCAN', mi.Number),
#                ('agg', 'AGGLOMERATIVE', mi.Number),
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
                ('avg_idle', 'Idle Wait', mi.Number),
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
        

        #register result tables in list of results 
        if self._mi_mode: #LAMI mode
            with open('/home/azhari/FROM_UBUNTU/runtime-EclipseApplication/vm_analysis/.tracing/folder_list.txt') as listF:
                folders = listF.readlines()
                folders = list(set(folders)) #remove duplicates
                folders = [fl.replace('\n','') for fl in folders] #remove newline at the end
                #folders = [fl.replace('(','\(') for fl in folders] #inset escape character
                #folders = [fl.replace(')','\)') for fl in folders] #inset escape character
                d = {}
                avgvec = {}
                fvec = {}
                #TODO compute per tracefile result tables
                for folder in folders:
                    traceName = folder.split('/')[-2] #extract name of tracefile
                    avgFileName = folder + 'avgdur.vector'
                    freqFileName = folder + 'frequency.vector'
                    with open(avgFileName,'r') as avgF, open(freqFileName,'r') as freqF:
                        #returns dictionary with key = VMID/PID and values = wait times and frequencies all in one list
                        d, avgvec, fvec = vectorize(avgF,freqF,traceName,d,avgvec,fvec)
                        #no clustering just output feature vectors
                        feature_vector_table = self._get_feature_vector_result_table(period_data,begin_ns, end_ns, traceName, d, avgvec, fvec)
                        self._mi_append_result_table(feature_vector_table)
                        #TODO call clustering function
                        #select clustering algorithms and parameters (define an enum for this)
                        alg_list = [Clustering.KMEANS]#, Clustering.KMEANS, Clustering.KMEANS] 
                        names, clusters = get_clusters(traceName,d,avgvec,fvec,alg_list)
                        #print(names,clusters)
                        feature_vector_table = self._get_clustering_result_table(period_data,begin_ns, end_ns, avgvec, fvec, names, clusters)
                        self._mi_append_result_table(feature_vector_table)
                        
                #TODO add aggregate result table
                feature_vector_table = self._get_feature_vector_result_table(period_data,begin_ns, end_ns, '', d, avgvec, fvec)
                self._mi_append_result_table(feature_vector_table)             
                #TODO call clustering on aggregate features
                #names, clusters = get_clusters('',d,avgvec,fvec,alg_list)               
                #feature_vector_table = self._get_clustering_result_table(period_data,begin_ns, end_ns, avgvec, fvec, names, clusters)
                #self._mi_append_result_table(feature_vector_table)             
                
                #TODO add clustering metrics result table
        

        else: #non LAMI mode
            #self._print_date(begin_ns, end_ns)
            self._print_feature_vector(feature_vector_table)

    #this is called after analysis is finished by the base class Command to create a summary table if required
    #to be overridden by subclasses 
    #implements code for populating optional result tables
    def _create_summary_result_tables(self): 
        self._mi_clear_result_tables() #no summary result table needed just clear all result tables 

    #now define all result table computing functions which were called in _analysis_tick() LAMI Mode
    def _get_feature_vector_result_table(self, period_data, begin_ns, end_ns, traceName, d, avgvec, fvec):
        result_table = \
            self._mi_create_result_table(self._MI_TABLE_CLASS_FEATURE_VECTOR,
                                         begin_ns, end_ns)

        if traceName == '': #aggregate of all traces
            cr3_list = d.keys()
        else:#filter out this traceName and take related VM/CR3 values and put in list
            cr3_list = [s for s in d.keys() if s.split('/')[0] == traceName]
        
        for vmpid in cr3_list:#iterate over all VMID/PIDs
            tr_name = vmpid.split('/')[0]
            vmid_cr3 = vmpid.split('/')[1]+'/'+vmpid.split('/')[2]
            result_table.append_row(
                #path         = mi.String(self._args.path),
                name         = mi.String(tr_name),
                vmcr3        = mi.String(vmid_cr3),
#                cluster      = mi.Number(int(km.labels_[i])),
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
                avg_idle     = mi.Number(avgvec[vmpid][7]),
                freq_idle    = mi.Number(fvec[vmpid][7])
            )

        return result_table

    def _get_clustering_result_table(self, period_data, begin_ns, end_ns, avgvec, fvec, names, clusters):
        result_table = \
            self._mi_create_result_table(self._MI_TABLE_CLASS_CLUSTERS,
                                         begin_ns, end_ns)
        
        i=0
        for vmpid in names:#iterate over all VMID/PIDs
            tr_name = vmpid.split('/')[0]
            vmid_cr3 = vmpid.split('/')[1]+'/'+vmpid.split('/')[2]
            result_table.append_row(
                #path         = mi.String(self._args.path),
                name         = mi.String(tr_name),
                vmcr3        = mi.String(vmid_cr3),
                km           = mi.Number(int(clusters['KMEANS'][0][i])),
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
                avg_idle     = mi.Number(avgvec[vmpid][7]),
                freq_idle    = mi.Number(fvec[vmpid][7])
            )
            i += 1

        return result_table

    #implement new analysis here ...
    #def _get_someanalysis_result_table(self, period_data, begin_ns, end_ns):

    #now define all result printing functions which were called in _analysis_tick() Non LAMI Mode
#    def _print_feature_vector(self, result_table):
#        row_format = '  {:<25} {:>10} ' #{:>10}  {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
#        label_header = row_format.format('VM/Process', 'Timer Duration') #, 'Timer Freq.', 'Disk Duration', 'Disk Freq.', 'Net Duration', 'Net Freq.', 'Task Duration', 'Task Freq.', 'Unknown Duration', 'Unknown Freq.', 'Non Root Duration', 'Non Root Freq.', 'Root Duration', 'Root Freq.')

#        def format_label(row):
#            return row_format.format(
#                '%s ' % (row.vmcr3.value),
#                row.cluster.value
#                row.freq_timer.value,
#                row.avg_disk.value,
#                row.freq_disk.value,
#                row.avg_net.value,
#                row.freq_net.value,
#                row.avg_task.value,
#                row.freq_task.value,
#                row.avg_unknown.value,
#                row.freq_unknown.value,
#                row.avg_nonroot.value,
#                row.freq_nonroot.value,
#                row.avg_root.value,
#                row.freq_root.value
#            )


#        graph = termgraph.BarGraph(
#            title='Cluster Labels',
#            unit='%',
#            get_value=lambda row: row.cluster._to_native_object()['value'],
#            get_label=format_label,
#            label_header=label_header,
#            data=result_table.rows
#        )

#        graph.print_graph()

def vectorize(avgF,freqF,traceName,d,avgvec,fvec):
    avglines = avgF.readlines()
    freqlines = freqF.readlines()
    avglines = [a.replace('\n','') for a in avglines]
    freqlines = [a.replace('\n','') for a in freqlines]
    #create a dictionary with key = traceFileName/VMID/PID and values the wait times and frequencies all in one list
    tmpd={traceName+'/'+avglines[i].split(',')[0] : avglines[i].split(',')[1:] + freqlines[i].split(',')[1:] for i in range(0,len(freqlines))}
    d.update(tmpd)
            
    for vmpid in tmpd.keys():
        f = np.zeros(8)
        avg = np.zeros(8)
        avg[0] = int(tmpd[vmpid][0]) #timer avg
        avg[1] = int(tmpd[vmpid][1]) 
        avg[2] = int(tmpd[vmpid][2]) 
        avg[3] = int(tmpd[vmpid][3]) 
        avg[4] = int(tmpd[vmpid][4]) 
        avg[5] = int(tmpd[vmpid][5]) 
        avg[6] = int(tmpd[vmpid][6]) 
        avg[7] = int(tmpd[vmpid][7]) 
        f[0] = int(tmpd[vmpid][8]) #timer freq
        f[1] = int(tmpd[vmpid][9])
        f[2] = int(tmpd[vmpid][10])
        f[3] = int(tmpd[vmpid][11])
        f[4] = int(tmpd[vmpid][12])
        f[5] = int(tmpd[vmpid][13])
        f[6] = int(tmpd[vmpid][14])
        f[7] = int(tmpd[vmpid][15])
        avgvec[vmpid] = avg
        fvec[vmpid] = f
          
    return d, avgvec, fvec  

def get_clusters(traceName,d,avgvec,fvec,alg_list):
    if traceName == '': #aggregate of all traces
        vmpid_list = d.keys()
    else:#filter out this traceName and take related VM/CR3 values and put in list
        vmpid_list = [s for s in d.keys() if s.split('/')[0] == traceName]
        
    samples = np.zeros((len(vmpid_list),8))
    i = 0
    #print(fvec)
    for vmpid in vmpid_list:
        samples[i,:] = fvec[vmpid]
        i += 1
    #TODO perform feature vector normalization
    #TODO build final feature vectors: avg | freq | avg+freq

    cl = {}
    #populate cl = {'KMEANS':(clusterlabels[1,2,3,1], paramlist[]), ...}
    #iterate over list of clustering algorithms
    for alg in alg_list: 
        func = Clustering.switcher.value.get(alg.value)
        c, param = func(samples)
        cl[alg.name] = (c,param)
        
    return vmpid_list, cl

    
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
