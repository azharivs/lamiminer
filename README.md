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

**Note:** You have to change the static path to the ```folder_list.txt``` file provided in the Python script ```vectorizer.py``` to point to your specific location.

```python
with open('/home/azhari/FROM_UBUNTU/runtime-EclipseApplication/vm_analysis/.tracing/folder_list.txt') as listF:
    folders = listF.readlines()
```

## Analysis Dependencies
The ```lttng-vectorizer``` analysis depends on the TraceCompass Incubator analysis ```VMblockVectorizerAnalysis``` which is maintained [here](https://github.com/Nemati/org.eclipse.tracecompass.incubator). You need to checkout the ```vahid``` branch to get the code.

Once the above analysis is added to TraceCompass you should create a new tracing project and import all trace files into it. Then create a VM Experiment for each trace file independently so that the ```VMblockVectorizer``` analysis is run producing ```avgdur.vector``` and ```frequency.vector``` files in the supplementary trace folders. These two files contain the average and frequency of various VCPU wait states and run states for different VMID/CR3. In addition, a ```folder_list.txt``` file will also be created under the suplementary folder of the new TraceCompass project. This file is populated with a list of full paths for various experiments of this project and will be used by the ```lttng-vectorizer``` LAMI script to execute feature extraction (vectorization) and clustering on this list of traces through loading the ```.vector``` files in those folders.

**Note:** There is no need to run the external analysis for each and every trace file. Once run for any of the trace files it will automatically load all ```.vector``` files enlisted in ```folder_list.txt``` and will perform analysis on all of them. It will then produce a single output under the trace for which it was run. The output, however, contains results for **all** trace files. Also note that any duplicate paths in ```folder_list.txt``` will be automatically eliminated by the analysis.

## Command Line Options
Example:
```--top 3 --feature fti,fta,fdi,fne --norm l2```
Consideres frequency of timer,task,disk,network and takes those VM/CR3 with top 3 frequecies in any of these features and also computes a second order (l2) norm of the features before clustering.

```python
        ap.add_argument('-t','--top', type=int, default=0,
                        help='Limit samples to VMPID/CR3 among top n candidates for'
                        ' at least one of the selcted features (default = 0: include all)')
        ap.add_argument('-f','--feature', type=str, default='*',
                        help='Only include these features given as comma separated list:'
                        ' [f:frequency|w:wait time][ti:timer|ta:task|di:disk|ne:network|ot:other|no:non-root|ro:root|id:idle]'
                        'Example: fti,fta,fdi,fne only considers frequencies of timer,task,disk,network'
                        'Example: w*: include all average wait times'
                        '(default=*: include all)')
        ap.add_argument('-n','--norm', type=str, default='l2',
                        help='Normalizing method for feature vector: l1|l2 (default =l2)')
        ap.add_argument('-c','--algs', type=str, default='*',
                        help='Only include these features given as comma separated list:'
                        ' kmeans3,dbscan,aggmax,aggmin,aggavg (agg:agglomerative)'
                        ' When relevant, the number at the end indicates the number of clusters'
                        'Example: kmeans3,dbscan '
                        '   Run kmeans with 3 clusters and dbscan'
                        '(default=kmeans3)')
```
