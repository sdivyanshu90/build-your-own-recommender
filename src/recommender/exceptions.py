"""Typed application exceptions."""


class RecommenderError(Exception):
    """Base exception with a safe public message."""


class ConfigurationError(RecommenderError):
    """Configuration is invalid or inconsistent."""


class DataQualityError(RecommenderError):
    """Input data violates a required quality rule."""


class ArtifactError(RecommenderError):
    """An artifact is missing, corrupt, or unsafe."""


class CompatibilityError(ArtifactError):
    """Two versioned artifacts cannot safely be used together."""


class NotReadyError(RecommenderError):
    """A runtime dependency has not loaded successfully."""
