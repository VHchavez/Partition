"""
Partition
Fragment the shit out of thoat molecule
"""

# Add imports here
# from .wuyang import wuyang
from .partition import Partition

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
