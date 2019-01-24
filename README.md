# lamiminer
Trace mining Python scripts as LAMI analysis for TraceCompass External Analysis. This repository greately borrows code from [lttng-analysis](https://github.com/lttng/lttng-analyses) and adheres to the LAMI specification detailed [here](https://github.com/lttng/lami-spec/blob/master/lami.adoc). It is encouraged that users refer to [lttng-analysis](https://github.com/lttng/lttng-analyses) and developers refer to [LAMI documentation](https://github.com/lttng/lami-spec/blob/master/lami.adoc) for further information. 

## Installation
clone the repo onto your local drive

run the setup script as sudoer

```bash
sudo ./setup.py install
```

## Using The LAMI Analysis
In TraceCompass you need to add your python script as a new External Analysis. Note that this has to be done for the Trace and not the Experiment. In most cases the script will be installed in ```/usr/local/bin``` and you would simply need to enter the script with absolute path in the menu for "Add External Analysis". Note that the name of the script has an ```-mi``` at the end which indicates the machine interface version of the script as opposed to its command line version.

For the particular case of vectorizing and clustering VM workload you simply need to enter ```/usr/local/bin/lttng-vectorizer-mi``` with no command arguments. Remember to give it some name as well to be listed under External Analysis. Then just right click on the new analysis and select Run External Analysis.

## Analysis Dependencies
The ```lttng-vectorizer``` analysis depends on the TraceCompass Incubator analysis ```VMblockVectorizerAnalysis``` which is maintained [here](https://github.com/Nemati/org.eclipse.tracecompass.incubator). You need to checkout the ```vahid``` branch to get the code.

Once the above analysis is added to TraceCompass you should create a new tracing project and import all trace files into it. Then create a VM Experiment for each trace file independently so that the ```VMblockVectorizer``` analysis is run producing ```avgdur.vector``` and ```frequency.vector``` files in the supplementary trace folders. These two files contain the average and frequency of various VCPU wait states and run states for different VMID/CR3. In addition, a ```folder_list.txt``` file will also be created under the suplementary folder of the new TraceCompass project. This file is populated with a list of full paths for various experiments of this project and will be used by the ```lttng-vectorizer``` LAMI script to execute feature extraction (vectorization) and clustering on this list of traces through loading the ```.vector``` files in those folders.

**Note:** There is no need to run the external analysis for each and every trace file. Once run for any of the trace files it will automatically load all ```.vector``` files enlisted in ```folder_list.txt``` and will perform analysis on all of them. It will then produce a single output under the trace for which it was run. The output, however, contains results for **all** trace files. Also note that any duplicate paths in ```folder_list.txt``` will be automatically eliminated by the analysis.

## Command Line Options
Example:
``` --top 3 --feature fti,fdi,fta,fne,wti,wdi,wta,wne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8 ```

Consideres frequency of timer,disk,task,network and blocking times upon them and takes those VM/CR3 with top 3 values in any of these features and then computes kmeans clustering with number of clusters = 2,3,4,5,6,7,8

``` --top 5 --feature fti,fdi,fta,fne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8 ```

``` --top 5 --feature wti,wdi,wta,wne --algs kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8 ```

```python
        ap.add_argument('-l','--list', type=str, default='',
                        help='Absolute path and filename of folder_list.txt file containing paths to all folders containing .vector files.')
        ap.add_argument('-t','--top', type=int, default=0,
                        help='Limit samples to VMPID/CR3 among top n candidates for'
                        ' at least one of the selected features (default = 0: include all)')
        ap.add_argument('-p','--proc', type=str, default='',
                        help='Only include these features from the VM/Process data files given as comma separated list:'
                        '[f:frequency|w:wait time][ti:timer|ta:task|di:disk|ne:network|ot:other|no:non-root|ro:root|l0:l0 preemption]'
                        'Example: fti,fta,fdi,fne only considers frequencies of timer,task,disk,network'
                        'Example: w*: include all average wait times'
                        '(default=*: include all)')
        ap.add_argument('-v','--vcpu', type=str, default='',
                        help='Only include these features from the VM/vCPU data files given as comma separated list:'
                        '[f:frequency|w:wait time|p:preemption freq|e:exit freq]'
                        'Used with f,w prefix, e.g., fti,wl0: [ti:timer|ta:task|di:disk|ne:network|ot:other|no:non-root|ro:root|l0:l0 preemption]'
                        'Used with p prefix, e.g., pvm,pho: [vm:VM/VM|ho:VM/Host|pr:in VM process preemption freq.|th:in VM thread preemption freq.]'
                        'Used with e prefix, e.g., eN1.N2,... a dot separated list of exit codes follow'
                        'Example: fti,fta,fdi,fne only considers frequencies of timer,task,disk,network'
                        'Example: w*: include all average wait times'
                        'Example: p*: include all types of VM preemptions'
                        'Example: e*: include all exit codes'
                        '(default='': none of the vcpu features)')
        ap.add_argument('-n','--norm', type=str, default='l2',
                        help='Normalizing method for feature vector: l1|l2 (default =l2)')
        ap.add_argument('--rate', default=False, action='store_true',
                        help='Scale frequency feature w.r.t. total execution time (root+non-root) to obtain waiting rate (default =False)')
        ap.add_argument('-c','--algs', type=str, default='kmeans3',
                        help='Only include these clustering algorithms given as comma separated list:'
                        'kmeans3,dbscan,aggmax,aggmin,aggavg (agg:agglomerative)'
                        'When relevant, the number at the end indicates the number of clusters'
                        'Example: kmeans3,dbscan '
                        '    Run kmeans with 3 clusters and run dbscan'
                        '(default=kmeans3)')
```
