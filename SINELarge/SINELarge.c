//  Scalable incomplete network embedding for large graphs

//  The SINELarge.c code was built upon the word2vec.c from https://code.google.com/archive/p/word2vec/

//  Modifications Copyright (C) 2018 <daokunzhang2015@gmail.com>
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

long long node_attribute_hash_size = 1000000000; // Maximum 1G node attribute pairs

struct node_context
{
    long long cn;
    long long source, target;
};

struct node_attribute
{
    long long cn;
    long long node, attribute;
};

struct node_neighbor
{
    long long node;
    long long neighbor_size;
    long long *neighbors;
};

struct node_content
{
    long long node;
    long long content_size;
    long long *contents;
    long long *freqs;
};

struct network
{
    long long node_num;
    long long attribute_num;
    struct node_neighbor *node_neighbors;
    struct node_content *node_contents;
};

char graph_file[MAX_STRING], emb_file[MAX_STRING], time_file[MAX_STRING];

struct network graph;

struct node_context *node_context_list_temp1, *node_context_list_temp2, *node_context_list = NULL;
long long node_context_list_size = 0, node_context_list_temp1_size = 0, node_context_list_temp2_size = 0;
long long node_context_list_temp1_max_size = 20000000, node_context_list_max_size = 20000000;

struct node_attribute *node_attribute_list;
long long node_attribute_list_size = 0, node_attribute_list_max_size = 100000000;
//long long *node_attribute_hash;
long long *node_freq, *attribute_freq;

long long window_size = 10, walk_num = 40, walk_length = 100;
long long layer1_size = 256;
double alpha = 0.025, starting_alpha;
double *syn0, *syn1neg_context, *syn1neg_content, *expTable;
clock_t start, finish;

long long negative = 5;
const long long node_table_size = 1e8, attribute_table_size = 1e8;
long long *node_table = NULL, *attribute_table;

// Parameters for node context pair and node attribute pair sampling
long long *node_context_alias = NULL, *node_attribute_alias;
double *node_context_prob = NULL, *node_attribute_prob;

long long total_samples = 100;

//Add node attribute pair to the node attribute list
long long AddNodeAttributeToList(long long node, long long attribute, long long cn)
{
    node_attribute_list[node_attribute_list_size].node = node;
    node_attribute_list[node_attribute_list_size].attribute = attribute;
    node_attribute_list[node_attribute_list_size].cn = cn;
    node_attribute_list_size++;
    if (node_attribute_list_size >= node_attribute_list_max_size)
    {
        node_attribute_list_max_size += 100 * graph.attribute_num;
        node_attribute_list = (struct node_attribute *)
                              realloc(node_attribute_list, node_attribute_list_max_size * sizeof(struct node_attribute));
    }
    return node_attribute_list_size - 1;
}

void ReadGraph()
{
    FILE *fp;
    long long i, j, k, l;
    fp = fopen(graph_file, "r");
    fscanf(fp, "%lld%lld", &graph.node_num, &graph.attribute_num);
    graph.node_neighbors = (struct node_neighbor *)malloc(graph.node_num*sizeof(struct node_neighbor));
    graph.node_contents = (struct node_content *)malloc(graph.node_num*sizeof(struct node_content));
    node_freq = (long long *)malloc(graph.node_num * sizeof(long long));
    attribute_freq = (long long *)malloc(graph.attribute_num * sizeof(long long));
    //for (i = 0; i < graph.node_num; i++) node_freq[i] = 0;
    for (i = 0; i < graph.attribute_num; i++) attribute_freq[i] = 0;
    for (i = 0; i < graph.node_num; i++)
    {
        fscanf(fp, "%lld", &k);
        graph.node_neighbors[k].node = k;
        graph.node_contents[k].node = k;
        fscanf(fp, "%lld", &l);
        graph.node_neighbors[k].neighbor_size = l;
        graph.node_neighbors[k].neighbors = (long long *)malloc(l * sizeof(long long));
        for (j = 0; j < l; j++)
            fscanf(fp, "%lld", &graph.node_neighbors[k].neighbors[j]);
        fscanf(fp, "%lld", &l);
        graph.node_contents[k].content_size = l;
        graph.node_contents[k].contents = (long long *)malloc(l * sizeof(long long));
        graph.node_contents[k].freqs = (long long *)malloc(l * sizeof(long long));
        for (j = 0; j < l; j++)
        {
            fscanf(fp, "%lld%lld", &graph.node_contents[k].contents[j], &graph.node_contents[k].freqs[j]);
            AddNodeAttributeToList(k, graph.node_contents[k].contents[j], graph.node_contents[k].freqs[j]);
            attribute_freq[graph.node_contents[k].contents[j]] += graph.node_contents[k].freqs[j];
        }
    }
    fclose(fp);
}

void InitUnigramTableContext()
{
    long long a, i;
    double train_nodes_pow = 0.0, d1, power = 0.75;
    if (node_table == NULL)
        node_table = (long long *)malloc(node_table_size * sizeof(long long));
    for (a = 0; a < graph.node_num; a++) train_nodes_pow += pow(node_freq[a], power);
    i = 0;
    d1 = pow(node_freq[0], power) / train_nodes_pow;
    for (a = 0; a < node_table_size; a++)
    {
        node_table[a] = i;
        if (a / (double)node_table_size > d1)
        {
            i++;
            d1 += pow(node_freq[i], power) / train_nodes_pow;
        }
        if (i >= graph.node_num) i = graph.node_num - 1;
    }
}

void InitUnigramTableContent()
{
    long long a, i;
    double train_attributes_pow = 0.0, d1, power = 0.75;
    attribute_table = (long long *)malloc(attribute_table_size * sizeof(long long));
    for (a = 0; a < graph.attribute_num; a++) train_attributes_pow += pow(attribute_freq[a], power);
    i = 0;
    d1 = pow(attribute_freq[0], power) / train_attributes_pow;
    for (a = 0; a < attribute_table_size; a++)
    {
        attribute_table[a] = i;
        if (a / (double)attribute_table_size > d1)
        {
            i++;
            d1 += pow(attribute_freq[i], power) / train_attributes_pow;
        }
        if (i >= graph.attribute_num) i = graph.attribute_num - 1;
    }
}

void Aggregate()
{
    long long i = 0, j = 0, k = 0;
    long long start = 0;
    node_context_list_temp2 = node_context_list;
    node_context_list_temp2_size = node_context_list_size;
    node_context_list = (struct node_context *)malloc(node_context_list_max_size * sizeof(struct node_context));
    while (i < node_context_list_temp1_size && j < node_context_list_temp2_size)
    {
        if ( (node_context_list_temp1[i].source < node_context_list_temp2[j].source) ||
             (node_context_list_temp1[i].source == node_context_list_temp2[j].source &&
              node_context_list_temp1[i].target <= node_context_list_temp2[j].target) )
        {
            if (start == 0)
            {
                node_context_list[k].source = node_context_list_temp1[i].source;
                node_context_list[k].target = node_context_list_temp1[i].target;
                node_context_list[k].cn = node_context_list_temp1[i].cn;
                k++;
                start = 1;
            }
            else
            {
                if (node_context_list[k-1].source == node_context_list_temp1[i].source &&
                    node_context_list[k-1].target == node_context_list_temp1[i].target)
                    node_context_list[k-1].cn += node_context_list_temp1[i].cn;
                else
                {
                    node_context_list[k].source = node_context_list_temp1[i].source;
                    node_context_list[k].target = node_context_list_temp1[i].target;
                    node_context_list[k].cn = node_context_list_temp1[i].cn;
                    k++;
                }
            }
            i++;
        }
        else
        {
            if (start == 0)
            {
                node_context_list[k].source = node_context_list_temp2[j].source;
                node_context_list[k].target = node_context_list_temp2[j].target;
                node_context_list[k].cn = node_context_list_temp2[j].cn;
                k++;
                start = 1;
            }
            else
            {
                if (node_context_list[k-1].source == node_context_list_temp2[j].source &&
                    node_context_list[k-1].target == node_context_list_temp2[j].target)
                    node_context_list[k-1].cn += node_context_list_temp2[j].cn;

                else
                {
                    node_context_list[k].source = node_context_list_temp2[j].source;
                    node_context_list[k].target = node_context_list_temp2[j].target;
                    node_context_list[k].cn = node_context_list_temp2[j].cn;
                    k++;
                }
            }
            j++;
        }
        if (k >= node_context_list_max_size)
        {
            node_context_list_max_size += 100 * graph.node_num;
            node_context_list = (struct node_context *)
                            realloc(node_context_list, node_context_list_max_size * sizeof(struct node_context));
        }
    }
    while (i < node_context_list_temp1_size)
    {
        if (start == 0)
        {
            node_context_list[k].source = node_context_list_temp1[i].source;
            node_context_list[k].target = node_context_list_temp1[i].target;
            node_context_list[k].cn = node_context_list_temp1[i].cn;
            k++;
            start = 1;
        }
        else
        {
            if (node_context_list[k-1].source == node_context_list_temp1[i].source &&
            node_context_list[k-1].target == node_context_list_temp1[i].target)
            node_context_list[k-1].cn += node_context_list_temp1[i].cn;
            else
            {
                node_context_list[k].source = node_context_list_temp1[i].source;
                node_context_list[k].target = node_context_list_temp1[i].target;
                node_context_list[k].cn = node_context_list_temp1[i].cn;
                k++;
            }
        }
        i++;
        if (k >= node_context_list_max_size)
        {
            node_context_list_max_size += 100 * graph.node_num;
            node_context_list = (struct node_context *)
                            realloc(node_context_list, node_context_list_max_size * sizeof(struct node_context));
        }
    }
    while (j < node_context_list_temp2_size)
    {
        if (start == 0)
        {
            node_context_list[k].source = node_context_list_temp2[j].source;
            node_context_list[k].target = node_context_list_temp2[j].target;
            node_context_list[k].cn = node_context_list_temp2[j].cn;
            k++;
            start = 1;
        }
        else
        {
            if (node_context_list[k-1].source == node_context_list_temp2[j].source &&
                node_context_list[k-1].target == node_context_list_temp2[j].target)
                node_context_list[k-1].cn += node_context_list_temp2[j].cn;
            else
            {
                node_context_list[k].source = node_context_list_temp2[j].source;
                node_context_list[k].target = node_context_list_temp2[j].target;
                node_context_list[k].cn = node_context_list_temp2[j].cn;
                k++;
            }
        }
        j++;
        if (k >= node_context_list_max_size)
        {
            node_context_list_max_size += 100 * graph.node_num;
            node_context_list = (struct node_context *)
                            realloc(node_context_list, node_context_list_max_size * sizeof(struct node_context));
        }
    }
    node_context_list_size = k;
    if (node_context_list_temp2 != NULL)
        free(node_context_list_temp2);
}

void QuickSortNodeContext(int low, int high)
{
    int i;
    int last;
    long long temp;
    if (low < high)
    {
        last = low;
        for (i = low+1; i<=high; i++)
        {
            if ( (node_context_list_temp1[i].source < node_context_list_temp1[low].source) ||
                 (node_context_list_temp1[i].source == node_context_list_temp1[low].source &&
                  node_context_list_temp1[i].target < node_context_list_temp1[low].target) )
            {
                last++;
                temp = node_context_list_temp1[i].source;
                node_context_list_temp1[i].source = node_context_list_temp1[last].source;
                node_context_list_temp1[last].source = temp;

                temp = node_context_list_temp1[i].target;
                node_context_list_temp1[i].target = node_context_list_temp1[last].target;
                node_context_list_temp1[last].target = temp;

                temp = node_context_list_temp1[i].cn;
                node_context_list_temp1[i].cn = node_context_list_temp1[last].cn;
                node_context_list_temp1[last].cn = temp;
            }
        }

        temp = node_context_list_temp1[low].source;
        node_context_list_temp1[low].source = node_context_list_temp1[last].source;
        node_context_list_temp1[last].source = temp;

        temp = node_context_list_temp1[low].target;
        node_context_list_temp1[low].target = node_context_list_temp1[last].target;
        node_context_list_temp1[last].target = temp;

        temp = node_context_list_temp1[low].cn;
        node_context_list_temp1[low].cn = node_context_list_temp1[last].cn;
        node_context_list_temp1[last].cn = temp;

        QuickSortNodeContext(low, last - 1);
        QuickSortNodeContext(last + 1, high);
    }
}

void RandomWalk()
{
    long long j, k, r, s = 0;
    long long cur_node;
    long long *rand_walk_nodes = (long long *) malloc(walk_length * sizeof(long long));
    for (j = 0; j < graph.node_num; j++)
    {
        cur_node = j;
        node_freq[cur_node]++;
        rand_walk_nodes[0] = j;
        for (k = 1; k < walk_length; k++)
        {
            if(graph.node_neighbors[cur_node].neighbor_size==0)
                break;
            cur_node = graph.node_neighbors[cur_node].neighbors[rand()%graph.node_neighbors[cur_node].neighbor_size];
            node_freq[cur_node]++;
            rand_walk_nodes[k] = cur_node;
            for (r = 1; r <= window_size; r++)
            {
                if (k - r < 0) continue;
                node_context_list_temp1[s].source = rand_walk_nodes[k-r];
                node_context_list_temp1[s].target = rand_walk_nodes[k];
                node_context_list_temp1[s].cn = 1;
                s++;
                node_context_list_temp1[s].source = rand_walk_nodes[k];
                node_context_list_temp1[s].target = rand_walk_nodes[k-r];
                node_context_list_temp1[s].cn = 1;
                s++;
            }
        }
        if ( (j > 0 && j % 10000 == 0) || (j == graph.node_num - 1) )
        {
            node_context_list_temp1_size = s;
            QuickSortNodeContext(0, node_context_list_temp1_size - 1);
            Aggregate();
            s = 0;
        }
    }
    free(rand_walk_nodes);
}

void InitNet()
{
    long long a, b;
    unsigned long long next_random = 1;
    syn0 = (double *)malloc(graph.node_num * layer1_size * sizeof(double));
    for (a = 0; a < graph.node_num; a++)
        for (b = 0; b < layer1_size; b++)
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (double)65536) - 0.5) / layer1_size;
        }
    syn1neg_context = (double *)malloc(graph.node_num * layer1_size * sizeof(double));
    syn1neg_content = (double *)malloc(graph.attribute_num * layer1_size * sizeof(double));
    for (a = 0; a < graph.node_num; a++)
        for (b = 0; b < layer1_size; b++)
            syn1neg_context[a * layer1_size + b] = 0;
    for (a = 0; a < graph.attribute_num; a++)
        for (b = 0; b < layer1_size; b++)
            syn1neg_content[a * layer1_size + b] = 0;
    InitUnigramTableContent();
}

/* The alias sampling algorithm, which is used to sample an node context pair in O(1) time. */
void InitNodeContextAliasTable()
{
    long long k;
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    double *norm_prob;
    long long *large_block;
    long long *small_block;
    if (node_context_alias == NULL)
        node_context_alias = (long long *)malloc(node_context_list_size * sizeof(long long));
    else
        node_context_alias = (long long *)realloc(node_context_alias, node_context_list_size * sizeof(long long));
    if (node_context_prob == NULL)
        node_context_prob = (double *)malloc(node_context_list_size * sizeof(double));
    else
        node_context_prob = (double *)realloc(node_context_prob, node_context_list_size * sizeof(double));
    norm_prob = (double*)malloc(node_context_list_size * sizeof(double));
    large_block = (long long*)malloc(node_context_list_size * sizeof(long long));
    small_block = (long long*)malloc(node_context_list_size * sizeof(long long));
    for (k = 0; k != node_context_list_size; k++) sum += node_context_list[k].cn;
    for (k = 0; k != node_context_list_size; k++) norm_prob[k] = (double)node_context_list[k].cn * node_context_list_size / sum;
    for (k = node_context_list_size - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        node_context_prob[cur_small_block] = norm_prob[cur_small_block];
        node_context_alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    while (num_large_block) node_context_prob[large_block[--num_large_block]] = 1;
    while (num_small_block) node_context_prob[small_block[--num_small_block]] = 1;
    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long SampleANodeContextPair(double rand_value1, double rand_value2)
{
    long long k = node_context_list_size * rand_value1;
    return rand_value2 < node_context_prob[k] ? k : node_context_alias[k];
}

void InitNodeAttributeAliasTable()
{
    long long k;
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    double *norm_prob;
    long long *large_block;
    long long *small_block;
    node_attribute_alias = (long long *)malloc(node_attribute_list_size * sizeof(long long));
    node_attribute_prob = (double *)malloc(node_attribute_list_size * sizeof(double));
    norm_prob = (double*)malloc(node_attribute_list_size * sizeof(double));
    large_block = (long long*)malloc(node_attribute_list_size * sizeof(long long));
    small_block = (long long*)malloc(node_attribute_list_size * sizeof(long long));
    for (k = 0; k != node_attribute_list_size; k++) sum += node_attribute_list[k].cn;
    for (k = 0; k != node_attribute_list_size; k++) norm_prob[k] = (double)node_attribute_list[k].cn * node_attribute_list_size / sum;
    for (k = node_attribute_list_size - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        node_attribute_prob[cur_small_block] = norm_prob[cur_small_block];
        node_attribute_alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    while (num_large_block) node_attribute_prob[large_block[--num_large_block]] = 1;
    while (num_small_block) node_attribute_prob[small_block[--num_small_block]] = 1;
    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long SampleANodeAttributePair(double rand_value1, double rand_value2)
{
    long long k = node_attribute_list_size * rand_value1;
    return rand_value2 < node_attribute_prob[k] ? k : node_attribute_alias[k];
}

void TrainModel()
{
    long long c, d;
    long long node, context, attribute, cur_pair;
    long long count = 0, last_count = 0;
    long long l1, l2, target, label;
    unsigned long long next_random = 1;
    double rand_num0, rand_num1, rand_num2;
    double f, g;
    double *neu1e = (double *)calloc(layer1_size, sizeof(double));
    long long *split_point = (long long *)calloc(walk_num, sizeof(long long));
    long long split_point_id = 0;
    InitNet();
    alpha = starting_alpha;
    printf("Training file: %s\n", graph_file);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Dimension: %lld\n", layer1_size);
    printf("Initial Alpha: %f\n", alpha);

    split_point[0] = 0;
    for (c = 1; c < walk_num; c++)
        split_point[c] = total_samples / walk_num;
    for (c = 0; c < total_samples % walk_num; c++)
        split_point[c + 1] += 1;
    for (c = 2; c < walk_num; c++)
        split_point[c] += split_point[c - 1];

    srand((unsigned)time(NULL));
    if (node_attribute_list_size > (long long)RAND_MAX + 1)
        printf("Warning: RAND_MAX is not large enough to guarantee the correctness of node attribute pair sampling!\n");

    while (1)
    {
        if (count >= total_samples) break;
        if (count == split_point[split_point_id])
        {
            split_point_id++;
            for (c = 0; c < graph.node_num; c++)
                node_freq[c] = 0;
            if (node_context_list != NULL)
                free(node_context_list);
            node_context_list = NULL;
            node_context_list_size = 0;
            RandomWalk();
            node_context_list = (struct node_context *)
                realloc(node_context_list, node_context_list_size * sizeof(struct node_context));
            InitUnigramTableContext();
            InitNodeContextAliasTable();
            if (node_context_list_size > (long long)RAND_MAX + 1)
                printf("Warning: RAND_MAX is not large enough to guarantee the correctness of node context pair sampling!\n");
        }
        if (count - last_count > 10000)
        {
            last_count = count;
            printf("Alpha: %f Progress %.3lf%%%c", alpha, (double)count / (double)(total_samples + 1) * 100, 13);
            fflush(stdout);
            alpha = starting_alpha * (1 - (double)count / (double)(total_samples + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        rand_num0 = rand() / (RAND_MAX * 1.0 + 1);
        if (rand_num0 <= 0.5)
        {
            rand_num1 = rand() / (RAND_MAX * 1.0 + 1);
            rand_num2 = rand() / (RAND_MAX * 1.0 + 1);
            cur_pair = SampleANodeContextPair(rand_num1, rand_num2);
            node = node_context_list[cur_pair].source;
            context = node_context_list[cur_pair].target;
            l1 = node * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            for (d = 0; d < negative + 1; d++)
            {
                if (d == 0)
                {
                    target = context;
                    label = 1;
                }
                else
                {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = node_table[(next_random >> 16) % node_table_size];
                    if (target == context) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg_context[c + l2];
                if (f > MAX_EXP) f = 1;
                else if (f < -MAX_EXP) f = 0;
                else f = expTable[(int)((f + MAX_EXP) * EXP_TABLE_SIZE / MAX_EXP / 2)];
                g = (label - f) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_context[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg_context[c + l2] += g * syn0[c + l1];
            }
            for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
        else
        {
            rand_num1 = rand() / (RAND_MAX * 1.0 + 1);
            rand_num2 = rand() / (RAND_MAX * 1.0 + 1);
            cur_pair = SampleANodeAttributePair(rand_num1, rand_num2);
            node = node_attribute_list[cur_pair].node;
            attribute = node_attribute_list[cur_pair].attribute;
            l1 = node * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            for (d = 0; d < negative + 1; d++)
            {
                if (d == 0)
                {
                    target = attribute;
                    label = 1;
                }
                else
                {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = attribute_table[(next_random >> 16) % attribute_table_size];
                    if (target == attribute) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++)
                    f += syn0[l1 + c] * syn1neg_content[c + l2];
                if (f > MAX_EXP) f = 1;
                else if (f < -MAX_EXP) f = 0;
                else f = expTable[(int)((f + MAX_EXP) * EXP_TABLE_SIZE / MAX_EXP / 2)];
                g = (label - f) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_content[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg_content[c + l2] += g * syn0[c + l1];
            }
            for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
        count++;
    }
    free(node_context_list_temp1);
}

void Output()
{
    long long a, b;
    FILE *fp = fopen(emb_file, "w");
    for (a = 0; a < graph.node_num; a++)
    {
        for (b = 0; b < layer1_size; b++)
        {
            if (b < layer1_size - 1)
                fprintf(fp, "%lf ", syn0[a * layer1_size + b]);
            else
                fprintf(fp, "%lf\n", syn0[a * layer1_size + b]);
        }
    }
    fclose(fp);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++)
    {
        if (!strcmp(str, argv[a]))
        {
            if (a == argc - 1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    FILE *fp;
    if (argc == 1)
    {
        printf("---Scalable incomplete network embedding for large graphs---\n");
        printf("---The code and following instructions are built upon word2vec.c by Mikolov et al.---\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-graph <file>\n");
        printf("\t\tThe input <file> for network embedding\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting network embeddings\n");
        printf("\t-time <file>\n");
        printf("\t\tUse <file> to save the running time\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of learned dimensions; default is 256\n");
        printf("\t-window <int>\n");
        printf("\t\tWindow size for collecting node context pairs; default is 10\n");
        printf("\t-walknum <int>\n");
        printf("\t\tThe number of random walks starting from per node; default is 40\n");
        printf("\t-walklen <int>\n");
        printf("\t\tThe length of random walks; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int> Million; default is 100\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-graph", argc, argv)) > 0) strcpy(graph_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(emb_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-time", argc, argv)) > 0) strcpy(time_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-walknum", argc, argv)) > 0) walk_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-walklen", argc, argv)) > 0) walk_length = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) >0) total_samples = atoi(argv[i + 1]);

    total_samples = total_samples * 1000000;
    starting_alpha = alpha;
    node_context_list_temp1 = (struct node_context *)
        malloc( node_context_list_temp1_max_size * sizeof(struct node_context));
    node_attribute_list = (struct node_attribute *)malloc(node_attribute_list_max_size * sizeof(struct node_attribute));
    expTable = (double *)malloc((EXP_TABLE_SIZE + 1) * sizeof(double));
    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        expTable[i] = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    ReadGraph();
    node_attribute_list = (struct node_attribute *)
        realloc(node_attribute_list, node_attribute_list_size * sizeof(struct node_attribute));
    start = clock();
    InitNodeAttributeAliasTable();
    TrainModel();
    finish = clock();
    printf("Total time: %lf secs for learning node embeddings\n", (double)(finish - start) / CLOCKS_PER_SEC);
    printf("----------------------------------------------------\n");
    fp = fopen(time_file, "w");
    fprintf(fp, "Total time: %lf secs for learning node embeddings\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fclose(fp);
    Output();
    return 0;
}
