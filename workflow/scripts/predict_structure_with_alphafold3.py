#!/usr/bin/env python
"""Predict structure with AlphaFold3."""

import glob
import os
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake  # noqa: TC004

os.makedirs(os.path.join(os.environ['TMPDIR'], 'jax_cache'), exist_ok=True)

with tempfile.TemporaryDirectory() as tmp_dir:
    input_dir_abs = os.path.join(tmp_dir, snakemake.wildcards.name, 'input')
    temp_output_dir_abs = os.path.join(tmp_dir, snakemake.wildcards.name, 'output')

    os.makedirs(input_dir_abs)
    os.makedirs(temp_output_dir_abs)

    shutil.copy(snakemake.input.sequence, input_dir_abs)

    if hasattr(snakemake.input, 'unpaired_msas'):
        for msa in snakemake.input.unpaired_msas:
            shutil.copy(msa, input_dir_abs)

    if hasattr(snakemake.input, 'paired_msas'):
        for msa in snakemake.input.paired_msas:
            shutil.copy(msa, input_dir_abs)

    if hasattr(snakemake.input, 'templates'):
        for template_dir in snakemake.input.templates:
            directory_name = os.path.basename(template_dir)
            new_directory_path = os.path.join(input_dir_abs, directory_name)
            os.mkdir(new_directory_path)

            for template in glob.glob(os.path.join(template_dir, '*.cif')):
                shutil.copy(template, os.path.join(new_directory_path, os.path.basename(template)))

    subprocess.Popen(['/usr/bin/chmod', '+x', '-R', input_dir_abs])  # noqa: S603

    input_file_name = os.path.basename(snakemake.input.sequence)

    subprocess.run(  # noqa: S603
        [
            '/usr/bin/apptainer',
            'exec',
            '--nv',
            f'--cwd={os.getcwd()}',
            f'--bind={os.getcwd()}:{os.getcwd()}',
            f"--bind={os.path.join(os.environ['TMPDIR'], 'jax_cache')}:/root/jax_cache",
            f'--bind={input_dir_abs}:/root/input',
            f'--bind={temp_output_dir_abs}:/root/output',
            f'--bind={snakemake.params.alphafold3_weights_dir}:/root/models',
            'containers/alphafold3.sif',
            'python',
            'third_party/alphafold3/run_alphafold.py',
            '--norun_data_pipeline',
            '--jax_compilation_cache_dir=/root/jax_cache',
            f'--num_recycles={snakemake.params.num_recycles}',
            f'--json_path=/root/input/{input_file_name}',
            '--model_dir=/root/models',
            '--output_dir=/root/output',
        ],
        check=True,
    )

    output_dir = os.path.dirname(snakemake.output.structures[0])
    os.makedirs(output_dir, exist_ok=True)

    sanitized_output_name = snakemake.wildcards.name.lower().replace(' ', '_')
    for sample_num in range(5):
        prediction_dir = os.path.join(
            temp_output_dir_abs, sanitized_output_name, f'seed-{snakemake.wildcards.seed}_sample-{sample_num}'
        )

        shutil.copy(
            os.path.join(
                prediction_dir, f'{sanitized_output_name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}_model.cif'
            ),
            os.path.join(
                output_dir, f'{snakemake.wildcards.name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}_model.cif'
            ),
        )

        shutil.copy(
            os.path.join(
                prediction_dir,
                f'{sanitized_output_name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}_summary_confidences.json',
            ),
            os.path.join(
                output_dir,
                f'{snakemake.wildcards.name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}'
                '_summary_confidences.json',
            ),
        )

        shutil.copy(
            os.path.join(
                prediction_dir,
                f'{sanitized_output_name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}_confidences.json',
            ),
            os.path.join(
                output_dir,
                f'{snakemake.wildcards.name}_seed-{snakemake.wildcards.seed}_sample-{sample_num}_confidences.json',
            ),
        )
