"""
Configuration management package.

Provides application settings and environment configuration:
- Settings class for typed configuration management (Settings)
- Global settings instance (settings)
- Development mode flag (IS_DEV)
"""

from .settings import IS_DEV, Settings, settings

__all__ = ["Settings", "settings", "IS_DEV"]
