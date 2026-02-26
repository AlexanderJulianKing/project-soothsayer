"""Thin shim — re-exports from the shared client so every bench stays in sync."""
import importlib.util, os, sys

_shared_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'llm_client.py')
_spec = importlib.util.spec_from_file_location("_shared_llm_client", _shared_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export public names so `from llm_client import X` keeps working
API_KEY = _mod.API_KEY
APIError = _mod.APIError
get_llm_response = _mod.get_llm_response
MAX_RETRIES = _mod.MAX_RETRIES
INITIAL_RETRY_DELAY = _mod.INITIAL_RETRY_DELAY
