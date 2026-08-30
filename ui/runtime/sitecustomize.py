"""Install the UI-only fast decoder in EVOKE's persistent inference worker."""

from __future__ import annotations

import os
import sys


if os.environ.get("EVOKE_UI_FAST_VAE", "0") == "1" and any("infer_single.py" in argument for argument in sys.argv):
    from lighttae_hook import install

    install()

if os.environ.get("EVOKE_UI_PRELOAD_VIGEO", "0") == "1" and any("infer_single.py" in argument for argument in sys.argv):
    from vigeo_hook import install

    install()
