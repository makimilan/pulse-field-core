"""Pulse-Field core module."""

from .impulse import Impulse, ImpulseStep, ImpulseEncoder, ImpulseDecoder
from .compatibility import Node, CompatibilityField, NodeRegistry
from .cgw_graph import CGWGraph
from .invariants import InvariantChecker
from .router import Route, RouteCache, System1Router, System2Router, ReversibilityPair
from .crystals import Crystal, SuperpositionAdapter, CrystalRegistry
from .archive import Archive, ArchiveEntry
from .autoarchitect import Autoarchitect, Mutation
from .config import Config
from .logging import StructuredLogger, get_logger, configure_logger
from .runtime import Runtime

__all__ = [
    'Impulse', 'ImpulseStep', 'ImpulseEncoder', 'ImpulseDecoder',
    'Node', 'CompatibilityField', 'NodeRegistry',
    'CGWGraph',
    'InvariantChecker',
    'Route', 'RouteCache', 'System1Router', 'System2Router', 'ReversibilityPair',
    'Crystal', 'SuperpositionAdapter', 'CrystalRegistry',
    'Archive', 'ArchiveEntry',
    'Autoarchitect', 'Mutation',
    'Config',
    'StructuredLogger', 'get_logger', 'configure_logger',
    'Runtime',
]
