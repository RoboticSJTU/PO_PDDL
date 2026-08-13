"""Central access to all domain and problem generation prompts."""

from .catalog import PromptCatalog, PromptRef, load_prompt, prompt_catalog

__all__ = ["PromptCatalog", "PromptRef", "load_prompt", "prompt_catalog"]
