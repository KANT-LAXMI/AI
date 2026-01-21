"""
Semantic Kernel Package
Contains SK setup, plugins, and prompt management
"""

from .kernel_setup import setup_kernel, get_kernel_response
from .plugins import MedicalAnalysisPlugin, DiagnosisAssistantPlugin

__all__ = ['setup_kernel', 'get_kernel_response', 'MedicalAnalysisPlugin', 'DiagnosisAssistantPlugin']