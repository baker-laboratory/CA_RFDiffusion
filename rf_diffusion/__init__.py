import contextlib
import os

import rf2aa as _  # noqa needed for registration



with contextlib.suppress(ImportError):
    from icecream import ic
    ic.configureOutput(includeContext=True)

projdir = os.path.dirname(__file__)

