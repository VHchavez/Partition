"""
Unit and regression test for the partition package.
"""

# Import package, test suite, and other packages as needed
import partition
import pytest
import sys

def test_partition_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "partition" in sys.modules
