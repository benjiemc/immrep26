#!/usr/bin/env python
"""Extract confidence metrics from AF3 predictions."""

import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from tcr_pmhc_predicted_structures.apps._log import setup_logger
from tcr_pmhc_predicted_structures.confidence_scores import (
    compute_alpha_bridge_scores,
    extract_confidence_score,
    get_sequence_information,
)
from tcr_pmhc_predicted_structures.predictions import CHAIN_ORDER

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake  # noqa: TC004

logger = logging.getLogger()
setup_logger(logger, 'info')


def collect_summary_scores(summary_confidences: dict) -> dict:
    """Collect all values from summary score dictionary."""
    scores = {key: val for key, val in summary_confidences.items() if not isinstance(val, list)}

    for chain_name, chain_ptm in zip(CHAIN_ORDER, summary_confidences['chain_ptm'], strict=False):
        scores[chain_name + '_ptm'] = chain_ptm

    for chain_name, chain_ptm in zip(CHAIN_ORDER, summary_confidences['chain_iptm'], strict=False):
        scores[chain_name + '_iptm'] = chain_ptm

    for i, (chain_name_1, chain_iptms) in enumerate(
        zip(CHAIN_ORDER, summary_confidences['chain_pair_iptm'], strict=False)
    ):
        for chain_name_2, iptm in zip(CHAIN_ORDER[i + 1 :], chain_iptms[i + 1 :], strict=False):
            if chain_name_1 != chain_name_2:
                scores[chain_name_1 + '_' + chain_name_2 + '_iptm'] = iptm

    for chain_name_1, chain_pae_min in zip(CHAIN_ORDER, summary_confidences['chain_pair_pae_min'], strict=False):
        for chain_name_2, pae_min in zip(CHAIN_ORDER, chain_pae_min, strict=False):
            scores[chain_name_1 + '_' + chain_name_2 + '_pae_min'] = pae_min

    return scores


logger.info('Loading sequences from %s', snakemake.input.sequences)
with open(snakemake.input.sequences, 'r') as fh:
    sequences = json.load(fh)

logger.info('Processing sequences')
sequences_flat, chain_ids, sequence_ids, insert_codes = get_sequence_information(sequences)

logger.info('Loading summary confidences from %s', snakemake.input.summary_confidences)
scores = {}

with open(snakemake.input.summary_confidences, 'r') as fh:
    summary_confidences = json.load(fh)

logger.info('Collecting summary confidence scores')
scores = {**scores, **collect_summary_scores(summary_confidences)}

logger.info('Combining summary confidence scores')

scores['combined_pae_min'] = np.mean(
    [
        score
        for key, score in scores.items()
        if re.match(
            '((peptide|mhc_chain_1)_(tcr_alpha_chain|tcr_beta_chain))|'
            '((tcr_alpha_chain|tcr_beta_chain)_(peptide|mhc_chain_1))_pae_min',
            key,
        )
    ]
)
scores['combined_iptm'] = np.mean(
    [score for key, score in scores.items() if re.match('(peptide|mhc_chain_1)_tcr_(alpha|beta)_chain_iptm', key)]
)

logger.info('Loading raw confidence scores from %s', snakemake.input.confidences)
with open(snakemake.input.confidences, 'r') as fh:
    confidences = json.load(fh)

logger.info('Extracting TCR-pMHC PAE Score')
pae = np.array(confidences['pae'])
scores['tcr_pmhc_pae_score'] = np.mean(
    extract_confidence_score(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        pae,
        mhc_type='MH1',
        score_type='pae',
        portion='all',
    )
)

logger.info('Extracting CDR-pMHC PAE Score')
scores['cdr_pmhc_pae_score'] = np.mean(
    extract_confidence_score(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        pae,
        mhc_type='MH1',
        score_type='pae',
        portion='binding',
    )
)

logger.info('Extracting TCR-pMHC pLDDT Score')
atom_plddt = np.array(confidences['atom_plddts'])
scores['tcr_pmhc_plddt_score'] = np.mean(
    extract_confidence_score(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        atom_plddt,
        mhc_type='MH1',
        score_type='atom_plddt',
        portion='all',
    )
)

logger.info('Extracting CDR-pMHC pLDDT Score')
scores['cdr_pmhc_plddt_score'] = np.mean(
    extract_confidence_score(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        atom_plddt,
        mhc_type='MH1',
        score_type='atom_plddt',
        portion='binding',
    )
)

logger.info('Computring TCR-pMHC AlphaBridge score')
scores['tcr_pmhc_alphabridge_score'] = np.mean(
    compute_alpha_bridge_scores(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        atom_plddt,
        pae,
        mhc_type='MH1',
        portion='all',
    )
)
logger.info('Computring CDR-pMHC AlphaBridge score')
scores['cdr_pmhc_alphabridge_score'] = np.mean(
    compute_alpha_bridge_scores(
        sequences_flat,
        chain_ids,
        sequence_ids,
        insert_codes,
        atom_plddt,
        pae,
        mhc_type='MH1',
        portion='binding',
    )
)

logger.info('Writing confidence scores to %s', snakemake.output[0])
with open(snakemake.output[0], 'w') as fh:
    fh.write(','.join(scores.keys()))
    fh.write('\n')
    fh.write(','.join(str(value) for value in scores.values()))
    fh.write('\n')
