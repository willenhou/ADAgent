"""
Configuration module for ADAgent paths.
Provides project root directory and common path utilities.
"""
import os
from pathlib import Path

# Get the project root directory (parent of this file's directory)
# This file is in adagent/, so we go up one level to get the project root
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return _PROJECT_ROOT

def get_model_dir() -> Path:
    """Get the path to the model weights directory."""
    return _PROJECT_ROOT / "model-weights"

def get_temp_dir() -> Path:
    """Get the path to the temporary files directory."""
    return _PROJECT_ROOT / "temp"

def get_docs_dir() -> Path:
    """Get the path to the docs directory."""
    return _PROJECT_ROOT / "adagent" / "docs"

def get_tools_dir() -> Path:
    """Get the path to the tools directory."""
    return _PROJECT_ROOT / "adagent" / "tools"

def get_subtools_dir() -> Path:
    """Get the path to the subtools directory."""
    return _PROJECT_ROOT / "adagent" / "tools" / "subtools"

def get_child_output_path() -> Path:
    """Get the path to the child output JSON file."""
    return get_temp_dir() / "child_output.json"

def get_system_prompts_path() -> Path:
    """Get the path to the system prompts file."""
    return get_docs_dir() / "system_prompts.txt"

