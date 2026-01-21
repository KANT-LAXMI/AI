"""
Database Package
Contains schema, operations, and initialization logic
"""

from .schema import init_database
from .operations import DatabaseOperations

__all__ = ['init_database', 'DatabaseOperations']