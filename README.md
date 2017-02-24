# ICRAR_DIA


Description of the Algorithm: 

Assuming a DAG I am going to use the following quantities:
1. __Parameters__: a) DAG, b) Specifics of the EA solver, e.g. Ngen (generations), c) Npop (number of individuals in population) etc 
2. __thread_spead__: This is an array of length *Nnodes*, i.e. equal to the total number of nodes in the Graph. The point is that since Number_of_nodes >> Number_of_threads, I will use slices of the DAG, to assign in the threads. 
3. __node_volume__: Array of length *Nnodes*. It represents the volume of data (in GB) to be processed. 
3. __node_load__: This is an array of length *Nnodes*. It represents the work that needs to be done on this node. The processing time then, for node i, which is assigned thread j is:
4. <del>__processing_times__]: Matrix(*Nnodes*,*Nnodes*). Definition __processing_times__(i,j) = __node_load__(i)/__thread_speed__(j) . It represents the time required for thread j to process node i.</del> 
5. __thread_traffic__: Matrix (dense) of length (Nnodes,Nnodes). It represents the speed for transfering information from __thread__(i) to __thread__(j)
6. __transfer_time__ (i,j,k): Scalar. It represents the time required to transfer data from thread(i) to thread(j), assuming that in thread(i) has been assigned node(k). This will be evaluated on the fly (probably), or else it should be represented in a dense 3 dimensional tensor format. 
