"""
audio_sources.py
Module to manage and store all audio source paths for the ASR model project.
"""

from typing import List

class AudioSources:
    """Class to manage audio source file paths."""
    def __init__(self):
        self.sources: List[str] = []

    def add_source(self, path: str) -> None:
        """Add a new audio source path."""
        self.sources.append(path)

    def get_sources(self) -> List[str]:
        """Return the list of audio source paths."""
        return self.sources
