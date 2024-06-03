from Bio import SeqIO
import re
import pandas as pd
from tqdm import tqdm




def translate2digital_sequence(fasta_path, window_size=3, stride=3, mode='codon'):
    digital_sequences_dict = {} 

    codon_replacements = {
                        'AAA': '0', 'AAC': '1', 'AAG': '2', 'AAT': '3',
                        'ACA': '4', 'ACC': '5', 'ACG': '6', 'ACT': '7',
                        'AGA': '8', 'AGC': '9', 'AGG': '10', 'AGT': '11',
                        'ATA': '12', 'ATC': '13', 'ATG': '14', 'ATT': '15',
                        'CAA': '16', 'CAC': '17', 'CAG': '18', 'CAT': '19',
                        'CCA': '20', 'CCC': '21', 'CCG': '22', 'CCT': '23',
                        'CGA': '24', 'CGC': '25', 'CGG': '26', 'CGT': '27',
                        'CTA': '28', 'CTC': '29', 'CTG': '30', 'CTT': '31',
                        'GAA': '32', 'GAC': '33', 'GAG': '34', 'GAT': '35',
                        'GCA': '36', 'GCC': '37', 'GCG': '38', 'GCT': '39',
                        'GGA': '40', 'GGC': '41', 'GGG': '42', 'GGT': '43',
                        'GTA': '44', 'GTC': '45', 'GTG': '46', 'GTT': '47',
                        'TAA': '48', 'TAC': '49', 'TAG': '50', 'TAT': '51',
                        'TCA': '52', 'TCC': '53', 'TCG': '54', 'TCT': '55',
                        'TGA': '56', 'TGC': '57', 'TGG': '58', 'TGT': '59',
                        'TTA': '60', 'TTC': '61', 'TTG': '62', 'TTT': '63'
                        }
    

    base_replacements = {
        'A': '0', 'T': '1', 'G': '2', 'C': '3'
    }


    if mode == 'codon':
        replacements = codon_replacements
    elif mode == 'base':
        replacements = base_replacements
        window_size = 1
        stride = 1

    
    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        sequence = str(record.seq)

        sequence = re.sub('[^ATGC]', '', sequence)
        
        result = []
        for i in range(0, len(sequence) - window_size + 1, stride):
            subsequence = sequence[i:i + window_size]
            if len(subsequence) == window_size:
                digital_value = replacements.get(subsequence, "")
                result.append(digital_value)
        digital_sequences_dict[record.id] = result

    return digital_sequences_dict

def create_labeled_feature_vectors_with_sorted_codons(numerical_sequences, label,codon_freq1=int(64)):
    labeled_feature_vectors = []

    for seq_name, num_seq in tqdm(numerical_sequences.items()):
        codon_freq = [0] * codon_freq1

        for codon_index in num_seq:
            codon_index = int(codon_index)
            codon_freq[codon_index] += 1

        total_codons = sum(codon_freq)

        codon_freq = [freq / total_codons for freq in codon_freq]

        sorted_indices_and_freqs = sorted(enumerate(codon_freq), key=lambda x: x[1], reverse=True)
        sorted_indices = [item[0] for item in sorted_indices_and_freqs]
        sorted_freqs = [item[1] for item in sorted_indices_and_freqs]

        feature_vector = sorted_indices + sorted_freqs + [label]
        labeled_feature_vectors.append(feature_vector)


    return labeled_feature_vectors

def create_labeled_feature_vectors_with_sorted_codons6(numerical_sequences, label,codon_freq1=int(64)):
    labeled_feature_vectors = []
    codon_freq = [0] * codon_freq1
    for codon_index in numerical_sequences:
        codon_index = int(codon_index)
        codon_freq[codon_index] += 1
    total_codons = sum(codon_freq)
    codon_freq = [freq / total_codons for freq in codon_freq]
    sorted_indices_and_freqs = sorted(enumerate(codon_freq), key=lambda x: x[1], reverse=True)
    sorted_indices = [item[0] for item in sorted_indices_and_freqs]
    sorted_freqs = [item[1] for item in sorted_indices_and_freqs]
    feature_vector = sorted_indices + sorted_freqs + [label]
    labeled_feature_vectors.append(feature_vector)

    return labeled_feature_vectors

def split_dict_by_half(digital_validationset):

    keys = list(digital_validationset.keys())
    midpoint = len(keys) // 2

    virus_keys = keys[:midpoint]
    host_keys = keys[midpoint:]

    virus_dict = {key: digital_validationset[key] for key in virus_keys}
    host_dict = {key: digital_validationset[key] for key in host_keys}

    return virus_dict, host_dict


def combine_and_convert_to_dataframe(virus_list, host_list):
    combined_list = virus_list + host_list
    df = pd.DataFrame(combined_list)
    return df










