import numpy as np 
import random
import re
from itertools import cycle, islice


def random_integers_sum_to_k(n, k):
    """
    Generate a list of `n` non-negative integers that sum up to `k`, uniformly at random.
    
    Parameters:
    n (int): Number of integers to generate.
    k (int): The sum of the generated integers.
    
    Returns:
    list: A list of `n` integers that sum up to `k`.
    """
    if n <= 0 or k < 0:
        raise ValueError("Number of integers must be positive and sum must be non-negative.")
    
    # Generate `n-1` random breakpoints and sort them
    breakpoints = np.sort(np.random.randint(0, k + 1, size=n - 1))
    
    # Add 0 at the beginning and `k` at the end to complete the intervals
    breakpoints = np.concatenate(([0], breakpoints, [k]))
    
    # Compute the differences between consecutive breakpoints
    result = np.diff(breakpoints)
    
    return result.tolist()

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def xor(a, b):
    return bool(a) != bool(b)

def join_consecutive_motifs(motifs):
    out = []
    current_motif = []
    for m in motifs:
        is_motif = re.compile('[A-Z]+(\d+)-(\d+)').match(m)
        is_intercalation = re.compile('(\d+)-(\d+)').match(m)
        assert xor(is_motif, is_intercalation), f'{m=}'
        if is_motif:
            current_motif.append(m)
        else:
            if current_motif:
                out.append(','.join(current_motif))
                current_motif = []
            out.append(m)
    return out

def get_contig_length(contig):
    contig_length = 0
    for m in contig.split(','):
        match = re.compile('[A-Z]+(\d+)-(\d+)').match(m)
        contig_start = int(match[1])
        contig_end = int(match[2])
        contig_length += contig_end - contig_start + 1
    assert contig_length > 0
    return contig_length

def normalize_contig(motifs):
    current_motif = []
    for m in motifs:
        is_motif = bool(re.compile('[A-Z]+(\d+)-(\d+)').fullmatch(m))
        is_intercalation = bool(re.compile('(\d+)-(\d+)').fullmatch(m))
        is_malformed_intercalation = bool(re.compile('(\d+)').fullmatch(m))
        print(f'{m=}: {is_motif=}, {is_intercalation=}, {is_malformed_intercalation=}')
        assert sum([is_motif, is_intercalation, is_malformed_intercalation]) == 1, f'{m=}: {is_motif=}, {is_intercalation=}, {is_malformed_intercalation=}'

        if is_malformed_intercalation:
            current_motif.append(f'{m}-{m}')
        else:
            current_motif.append(m)
    return current_motif

def shuffle_contig_string(
        contig_list,
        length_min,
        length_max,
):
    '''
    Shuffle a contig string.
    Parameters:
        contig_list (list): example: [10-30,'A1-20',10-20,'A30-40',10-30]
    '''

    print(f'shuffle_and_random_partition START: {contig_list=}')
    length = (length_min, length_max)
    print(f'{length=}')
    length_int = random.randrange(length[0], length[1])
    length = f'{length_int}-{length_int}'

    shuffled_contig_list = contig_list
    print(f'before shuffle: {contig_list=}')
    print(f'{shuffled_contig_list=}')

    shuffled_contig_list = normalize_contig(shuffled_contig_list)

    # Join consecutive motif regions.
    shuffled_contig_list = np.array(join_consecutive_motifs(shuffled_contig_list))
    print('after join_consecutive_motifs:')
    print(f'{shuffled_contig_list=}')

    is_motif = np.array([e[0].isalpha() for e in shuffled_contig_list])
    motifs = shuffled_contig_list[is_motif]
    np.random.shuffle(motifs)

    n_motif_residues = 0
    for m in motifs:
        n_motif_residues += get_contig_length(m)
            
    n_intercalations = len(motifs) + 1
    l_total_intercalations = length_int - n_motif_residues
    intercalations = random_integers_sum_to_k(n_intercalations, l_total_intercalations)
    print(f'{n_intercalations=} {l_total_intercalations=}{intercalations=}')

    intercalation_strings = [f'{i}-{i}' for i in intercalations]
    contig_list = roundrobin(intercalation_strings, motifs)
    print(f'shuffle_and_random_partition END:   {contig_list=}')
    return contig_list, length

