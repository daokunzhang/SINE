# SINE: Sclable Incomplete Network Embedding

Code for ICDM-2018 paper "SINE: Sclable Incomplete Network Embedding"

Authors: Daokun Zhang, Jie Yin, Xingquan Zhu and Chengqi Zhang

Contact: Daokun Zhang (daokunzhang2015@gmail.com)

This project is implemented in two versions:

1) The folder "SINE" is the initial version of SINE. It firstly conducts random walks and collects node context pairs though a hash table, then learns parameters with stochastic gradient descent by randomly sampling a node-context pair or node-attribute pair at each iteration.

	Please run the "SINERun.sh" file to run this implementation on cora and citeseer network.

2) The folder "SINELarge" is the version of SINE adapted for large networks with millions of nodes. This implementation operates random walk and parameter learning alternately. 

    Please run the "SINELargeRun.sh" file to run this implementation on cora and citeseer network.

The format of the input network is as following:

In the input file, the first line is the number of nodes ("node_num") and the number of node content attributes ("feat_num") separated by whitespace:

    node_num feat_num

From the second line, one by one, each node's id, neighbor list and content features are provided. The following is the information for a node:

    node_id
    neigh_num
    neigh_id1 neigh_id2 neigh_id3 ......
    nonzero_feat_num
    nonzero_attribute_id1 nonzero_attribute_val1 nonzero_attribute_id2 nonzero_attribute_val2 ......

Above, "node_id" is the id of current node. "neigh_num" is the number of neighbors of current node, and "neigh_id1, neigh_id2, neigh_id3..." is the neighbor list of current node. "nonzero_feat_num" is the number of observed nonzero feature values of the current node and "nonzero_attribute_id1 nonzero_attribute_val1 nonzero_attribute_id2 nonzero_attribute_val2 ......" is the nonzero feature value list of current node, where "nonzero_attribute_id1" is the id of the observed attribute in which current node take nonzero value and "nonzero_attribute_val1" is the corresponding attribute value.

Taking the "cora.txt" file as an example, its content is as follows:

    2708 1433
    0
    5
    8 14 258 435 544
    20
    118 1 125 1 176 1 252 1 351 1 456 1 507 1 521 1 619 1 648 1 698 1 702 1 734 1 845 1 902 1 1205 1 1209 1 1236 1 1352 1 1426 1
    1
    1
    344
    17
    12 1 509 1 620 1 763 1 882 1 893 1 978 1 1131 1 1135 1 1177 1 1207 1 1256 1 1263 1 1266 1 1332 1 1389 1 1425 1
    2
    4
    410 471 552 565
    22
    45 1 209 1 212 1 239 1 292 1 394 1 510 1 514 1 581 1 621 1 623 1 638 1 1075 1 1132 1 1177 1 1206 1 1263 1 1289 1 1349 1 1389 1 1415 1 1421 1
    3
    3
    197 463 601
    21
    41 1 93 1 99 1 149 1 594 1 617 1 624 1 648 1 874 1 915 1 942 1 988 1 1004 1 1049 1 1071 1 1170 1 1177 1 1194 1 1292 1 1348 1 1349 1
    ......

The options of SINE/SINELarge are as follows:

    -graph <file>
        The input <file> for network embedding
    -output <file>
        Use <file> to save the resulting network embeddings
    -time <file>
        Use <file> to save the running time
    -size <int>
        Set size of learned dimensions; default is 256
    -window <int>
        Window size for collecting node context pairs; default is 10
    -walknum <int>
        The number of random walks starting from per node; default is 40
    -walklen <int>
        The length of random walks; default is 100
    -negative <int>
        Number of negative examples; default is 5, common values are 3 - 10
    -alpha <float>
        Set the starting learning rate; default is 0.025
    -samples <int>
        Set the number of training samples as <int> Million; default is 100

If you find this project is useful, please cite this paper:

	@inproceedings{zhang2018sine,
  	    title={SINE: Scalable Incomplete Network Embedding},
  	    author={Zhang, Daokun and Yin, Jie and Zhu, Xingquan and Zhang, Chengqi},
  	    booktitle={IEEE International Conference on Data Mining},
  	    year={2018},
  	    organization={IEEE}
	}
