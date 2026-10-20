#! /usr/bin/env python
"""Process IMMREP dataset."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import anarci
import pandas as pd
import tidytcells

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

IMGT_CDR1: set[int] = set(range(27, 38 + 1))
IMGT_CDR2: set[int] = set(range(56, 65 + 1))
IMGT_CDR3: set[int] = set(range(105, 117 + 1))


def mhc_slug_to_code(slug: str) -> str:
    """Convert mhc slugs into the allele codes.

    >>> mhc_slug_to_code('hla_a_02_01')
    'HLA-A*02:01'

    >>> mhc_slug_to_code('h2_kb')
    'H2-Kb'

    TODO: Make this better for other mouse alleles
    """
    species = 'human' if slug.startswith('hla') else 'mouse' if slug.startswith('h2') else None

    if species == 'human':
        code = slug.upper()
        code = code.split('_')
        code = code[0] + '-' + code[1] + '*' + ':'.join(code[2:])

    elif species == 'mouse':
        code = slug.title()
        code = code.replace('_', '-')

    else:
        code = slug

    return code


def assign_species(allele: str) -> str | None:
    """Assign species based on MHC allele code."""
    if re.search('^HLA-', allele):
        return 'Human'

    if re.search('^H2-?', allele):
        return 'Mouse'

    if re.search('^Gaga', allele):
        return 'Chicken'

    return None


def get_cdrs(sequence: str) -> tuple[str, str, str]:
    """Extract CDRs from full TCR chain sequence."""
    numbered_sequence = anarci.number(sequence)

    cdr1 = []
    cdr2 = []
    cdr3 = []

    for (seq_id, _), residue in numbered_sequence[0]:
        if residue == '-':
            continue

        if seq_id in IMGT_CDR1:
            cdr1.append(residue)

        elif seq_id in IMGT_CDR2:
            cdr2.append(residue)

        elif seq_id in IMGT_CDR3:
            cdr3.append(residue)

    return ''.join(cdr1), ''.join(cdr2), ''.join(cdr3)


logger.info('Loading dataset from %s', snakemake.input.dataset)
dataset = pd.read_csv(snakemake.input.dataset)

logger.info('Adding MHC information')
mhc_sequences = pd.concat(
    [pd.read_json(str(path), orient='index') for path in Path(snakemake.input.mhc_sequences).iterdir()]
)

mhc_sequences['mhc_code'] = mhc_sequences.index.map(mhc_slug_to_code)
mhc_sequences['species'] = mhc_sequences['mhc_code'].map(assign_species)

mhc_sequences.loc[mhc_sequences['species'] == 'Human', 'mhc_code'] = mhc_sequences.loc[
    mhc_sequences['species'] == 'Human', 'mhc_code'
].apply(tidytcells.mh.standardize, species='homosapiens')

mhc_sequences.loc[mhc_sequences['species'] == 'Mouse', 'mhc_code'] = mhc_sequences.loc[
    mhc_sequences['species'] == 'Mouse', 'mhc_code'
].apply(tidytcells.mh.standardize, species='musmusculus')

dataset['mhc_allele'] = dataset['hla_long'].map(tidytcells.mh.standardise)

dataset = (
    dataset.merge(
        mhc_sequences[['mhc_code', 'canonical_sequence']], how='left', left_on='mhc_allele', right_on='mhc_code'
    )
    .rename({'canonical_sequence': 'mhc_chain_1'}, axis='columns')
    .drop('mhc_code', axis='columns')
)

B2M_SEQUENCES = {
    'Human': 'IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM',
    'Mouse': 'IQKTPQIQVYSRHPPENGKPNILNCYVTQFHPPHIEIQMLKNGKKIPKVEMSDMSFSKDWSFYILAHTEFTPTETDTYACRVKHASMAEPKTVYWDRDM',
}

dataset['mhc_chain_2'] = B2M_SEQUENCES['Human']

logger.info('Renaming epitope column')
dataset = dataset.rename({'epitope_aa': 'peptide'}, axis='columns')

logger.info('Cleaning TCR gene codes')
dataset[['v_alpha', 'j_alpha', 'v_beta', 'j_beta']] = dataset[['TRAV_IMGT', 'TRAJ_IMGT', 'TRBV_IMGT', 'TRBJ_IMGT']].map(
    tidytcells.tr.standardise
)

logger.info('Renaming full TCR sequences')
dataset = dataset.rename({'full_seq_alpha_aa': 'tcr_alpha_chain', 'full_seq_beta_aa': 'tcr_beta_chain'}, axis='columns')

logger.info('Getting CDRs')
dataset[['cdr1_alpha', 'cdr2_alpha', 'cdr3_alpha']] = dataset['tcr_alpha_chain'].map(get_cdrs).apply(pd.Series)
dataset[['cdr1_beta', 'cdr2_beta', 'cdr3_beta']] = dataset['tcr_beta_chain'].map(get_cdrs).apply(pd.Series)

logger.info('Outputting to %s', snakemake.output[0])
dataset[
    [
        'tcr_id',
        'v_alpha',
        'j_alpha',
        'v_beta',
        'j_beta',
        'cdr1_alpha',
        'cdr2_alpha',
        'cdr3_alpha',
        'cdr1_beta',
        'cdr2_beta',
        'cdr3_beta',
        'tcr_alpha_chain',
        'tcr_beta_chain',
        'peptide',
        'mhc_allele',
        'mhc_chain_1',
        'mhc_chain_2',
    ]
].to_csv(snakemake.output[0], index=False)
