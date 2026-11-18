import os


def _ensure_torch_lib_in_ld_library_path():
    """
    Some prebuilt CUDA extensions in this repo were built with an RPATH that
    doesn't exist on all servers/containers. Ensuring torch's lib directory is
    on LD_LIBRARY_PATH makes libs like libc10.so discoverable at import time.
    """
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isdir(torch_lib):
            return

        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in ld_library_path.split(":") if p] if ld_library_path else []
        if torch_lib in parts:
            return

        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{ld_library_path}" if ld_library_path else torch_lib
    except Exception:
        return


_ensure_torch_lib_in_ld_library_path()
