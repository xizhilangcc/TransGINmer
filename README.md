# TransGINmer

## Description

TransGINmer is a python library for identifying virus from metagenomic data.It is constructed by a multi-head self-attention mechanism and a Graph Isomorphism Networks (GIN). The multi-head self-attention mechanism is used to transform sequence structures into graph structures and captures the global dependencies between codon tokens. GIN is used to further extract sequence features in order to identify viral sequences.

## Dependences

To utilize TransGINmer on Linux, the following Python packages are needed to be previously installed.

- Bio
- re
- tqdm
- pandas
- torch
- argparse
- math
- torch_geometric
- numpy
- sklearn
- metric
- os

## Usage

When using TransGINmer, user need change the name of the input FASTA file to contig.fasta and execute the following Python statement. Then the predictions.txt result file will be generated.

'python prediction.py -dataset contig.fasta'
