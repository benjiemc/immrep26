#! /usr/bin/env python
"""Calculate pairwise distance matrix of the sequence distances between CDRs."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake  # noqa: TC004

LOG_MAP = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}

logger = logging.getLogger()

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
logger.addHandler(handler)

logger.setLevel(LOG_MAP[snakemake.config['log_level']])

logger.info('Load sequences from %s', snakemake.input[0])
dataset = pd.read_csv(snakemake.input[0], index_col='tcr_id')

logger.info('Extract %s columns', snakemake.wildcards.cdr)
sequences = dataset[snakemake.wildcards.cdr].to_numpy()

logger.info('Calculating distances')
num_tcrs = len(sequences)

distance_matrix = np.zeros((num_tcrs, num_tcrs), dtype=int)
for j in range(num_tcrs - 1):
    for i in range(j + 1, num_tcrs):
        distance_matrix[i, j] = damerau_levenshtein_distance(sequences[i], sequences[j])

distance_matrix = np.max([distance_matrix, distance_matrix.T], axis=0)

logger.info('Outputting distance matrix to %s', snakemake.output[0])
np.savetxt(snakemake.output[0], distance_matrix, fmt='%i')
