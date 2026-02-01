"""SLURM wrapper for GPU gradient fitting jobs.

Submits gradient fitting as SLURM batch jobs when GPU is needed.
Falls back to CPU fitting on the login node for small jobs.
"""
import os
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=grounded-fit-{job_id}
#SBATCH --output={log_dir}/fit_{job_id}.out
#SBATCH --error={log_dir}/fit_{job_id}.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G

cd {work_dir}
python -c "
import sys
sys.path.insert(0, '{src_dir}')
sys.path.insert(0, '{shinka_dir}')

import json
import numpy as np
from gradient.fitter import fit_expression

data = json.load(open('{data_path}'))
inputs = np.array(data['inputs'])
targets = np.array(data['targets'])
expr_str = data['expr_str']

result = fit_expression(expr_str, inputs, targets)
if result is not None:
    output = {{
        'template_name': result.template_name,
        'params': result.params.tolist(),
        'param_names': result.param_names,
        'train_mse': result.train_mse,
    }}
else:
    output = None

json.dump(output, open('{result_path}', 'w'))
"
"""


def submit_fit_job(expr_str: str, inputs, targets,
                   job_id: str = "0",
                   log_dir: str = "/tmp/grounded_fit") -> Optional[str]:
    """Submit a gradient fitting job to SLURM.

    Args:
        expr_str: expression to fit
        inputs: input data array
        targets: target values array
        job_id: unique job identifier
        log_dir: directory for SLURM logs
    Returns:
        SLURM job ID string, or None if submission failed
    """
    import numpy as np

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    src_dir = str(Path(__file__).resolve().parent.parent)
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    shinka_dir = str(Path(repo_root) / "ShinkaEvolve")

    # Save data
    data_path = str(log_dir_path / f"fit_data_{job_id}.json")
    result_path = str(log_dir_path / f"fit_result_{job_id}.json")
    with open(data_path, 'w') as f:
        json.dump({
            'expr_str': expr_str,
            'inputs': np.asarray(inputs).tolist(),
            'targets': np.asarray(targets).tolist(),
        }, f)

    # Write SLURM script
    script = SLURM_TEMPLATE.format(
        job_id=job_id, log_dir=str(log_dir_path),
        work_dir=src_dir, src_dir=src_dir, shinka_dir=shinka_dir,
        data_path=data_path, result_path=result_path,
    )
    script_path = str(log_dir_path / f"fit_script_{job_id}.sh")
    with open(script_path, 'w') as f:
        f.write(script)

    try:
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            slurm_id = result.stdout.strip().split()[-1]
            logger.info(f"Submitted SLURM fit job {slurm_id}")
            return slurm_id
        else:
            logger.warning(f"SLURM submission failed: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"SLURM submission error: {e}")
        return None


def load_fit_result(job_id: str, log_dir: str = "/tmp/grounded_fit") -> Optional[Dict]:
    """Load result from a completed SLURM fit job."""
    result_path = Path(log_dir) / f"fit_result_{job_id}.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path) as f:
            return json.load(f)
    except Exception:
        return None
