"""The Python source of a Kaggle notebook that runs one forge session.

The notebook carries the whole session folder inside itself, installs the torch version
the session needs, unpacks the folder and runs ``forge.session`` with the time that is left.
"""

from __future__ import annotations

import base64

# Time kept back from Kaggle's own limit, for unpacking and for writing the results.
NOTEBOOK_MARGIN_SECONDS = 60

TEMPLATE = '''"""A forge session: install torch, unpack the job and run it within the time left."""
import base64
import io
import os
import subprocess
import sys
import time
import zipfile

started = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--disable-pip-version-check", *{packages!r}], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode({bundle!r}))).extractall("forge")
left = {seconds} - (time.time() - started) - {margin}
subprocess.run([sys.executable, "-m", "forge.session", "forge", str(max(left, 1))],
    env=dict(os.environ, PYTHONPATH="forge"))
'''


def source(bundle: bytes, seconds: int, torch: str, requirements: list[str]) -> str:
    """Return the notebook's code for one session bundle."""
    packages = [f"torch=={torch}", "numpy", "safetensors", *requirements]
    return TEMPLATE.format(
        packages=packages,
        bundle=base64.b64encode(bundle).decode(),
        seconds=seconds,
        margin=NOTEBOOK_MARGIN_SECONDS,
    )
