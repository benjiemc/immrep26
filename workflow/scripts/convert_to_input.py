#! /usr/bin/env python
"""Convert sequence data into different formats for structure prediction methods."""

import glob
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING

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


def format_sequences_for_af3(
    sequences: dict,
    output_dir: str,
    seed: int,
    *,
    unpaired_msa_paths: list[str],
    paired_msa_paths: list[str],
    templates: dict[str, list[dict[str, str]]],
) -> None:
    """Create a JSON file to predict a structure with AlphaFold 3."""
    output_name = os.path.join(output_dir, f"{sequences['tcr_id']}_seed-{seed}.json")

    output = {
        'name': sequences['tcr_id'],
        'modelSeeds': [seed],
        'sequences': [
            {'protein': {'id': sequence_id, 'sequence': sequences[sequence_type]}}
            for sequence_id, sequence_type in (
                ('D', 'tcr_alpha_chain'),
                ('E', 'tcr_beta_chain'),
                ('C', 'peptide'),
                ('A', 'mhc_chain_1'),
                ('B', 'mhc_chain_2'),
            )
            if sequence_type in sequences
        ],
        'dialect': 'alphafold3',
        'version': 2,
    }

    new_sequence_entries = []
    for sequence_entry in output['sequences']:
        new_sequence_entry = {**sequence_entry}

        if sequence_entry['protein']['id'] == 'C':
            msa = f">query\n{sequence_entry['protein']['sequence']}\n"
            new_sequence_entry['protein']['unpairedMsa'] = msa
            new_sequence_entry['protein']['pairedMsa'] = msa
            new_sequence_entry['protein']['templates'] = []

        else:
            key = hashlib.sha256(sequence_entry['protein']['sequence'].encode()).hexdigest()[:8]
            relevant_unpaired_msas = [msa_path for msa_path in unpaired_msa_paths if key in msa_path]

            if len(relevant_unpaired_msas) < 1:
                logger.error(
                    'Missing unpaired MSAs for sequence: %s (key: %s)',
                    sequence_entry['protein']['sequence'],
                    key,
                )
                new_sequence_entry['protein']['unpairedMsa'] = ''

            else:
                new_sequence_entry['protein']['unpairedMsaPath'] = relevant_unpaired_msas[0]

            relevant_paired_msas = [msa_path for msa_path in paired_msa_paths if key in msa_path]

            if len(relevant_paired_msas) < 1:
                logger.warning(
                    'Missing paired MSAs for sequence: %s (key: %s)',
                    sequence_entry['protein']['sequence'],
                    key,
                )
                new_sequence_entry['protein']['pairedMsa'] = ''

            else:
                new_sequence_entry['protein']['pairedMsaPath'] = relevant_paired_msas[0]

            try:
                new_sequence_entry['protein']['templates'] = templates[key]

            except KeyError:
                logger.warning(
                    'Missing templates for sequence: %s (key: %s)',
                    sequence_entry['protein']['sequence'],
                    key,
                )
                new_sequence_entry['protein']['templates'] = []

        new_sequence_entries.append(new_sequence_entry)

    output['sequences'] = new_sequence_entries

    with open(output_name, 'w') as output_fh:
        logger.debug('Writing to %s', output_name)
        json.dump(output, output_fh)


logger.info('Loading sequences from %s', snakemake.input.sequences)
with open(snakemake.input.sequences, 'r') as fh:
    sequences = json.load(fh)

dirname = os.path.dirname(snakemake.input.unpaired_msas[0])
unpaired_msa_paths = [path.replace(dirname, '').strip('/') for path in snakemake.input.unpaired_msas]

dirname = os.path.dirname(snakemake.input.paired_msas[0])
paired_msa_paths = [path.replace(dirname, '').strip('/') for path in snakemake.input.paired_msas]

templates = {}
for path in snakemake.input.templates:
    name = os.path.basename(path)
    template_entries = defaultdict(dict)
    indices_paths = glob.glob(os.path.join(path, '*_indices.json'))
    structure_paths = glob.glob(os.path.join(path, '*.cif'))

    for indice_path in indices_paths:
        entry_name = re.search(r'(\d+)_indices.json', os.path.basename(indice_path)).group(1)
        template_entries[entry_name]['indices_path'] = indice_path

    for structure_path in structure_paths:
        entry_name = re.search(r'(\d+).cif', os.path.basename(structure_path)).group(1)
        template_entries[entry_name]['structure_path'] = structure_path

    template_entries = sorted(template_entries.items())

    entry_templates = []
    for _, entry in template_entries:
        with open(entry['indices_path'], 'r') as fh:
            entry_template = json.load(fh)

        dirname = os.path.dirname(path)
        entry_template['mmcifPath'] = entry['structure_path'].replace(dirname, '').strip('/')

        entry_templates.append(entry_template)

    templates[name] = entry_templates

output_dir = os.path.dirname(snakemake.output[0])

logger.info('Formatting sequences for AlphaFold 3 and outputting to %s', snakemake.output[0])
format_sequences_for_af3(
    sequences,
    output_dir,
    int(snakemake.wildcards.seed),
    unpaired_msa_paths=unpaired_msa_paths,
    paired_msa_paths=paired_msa_paths,
    templates=templates,
)
