"""Safe, provider-neutral agent errors."""


class AgentError(RuntimeError):
    """Base class for expected agent failures."""


class ConfigurationError(AgentError):
    """Raised when agent configuration is missing or invalid."""


class UnsupportedModelError(AgentError):
    """The selected model has no verified support for the required tool API."""


class ProviderRequestError(AgentError):
    """A provider request failed; details are intentionally redacted."""


class ProviderProtocolError(AgentError):
    """A provider returned a malformed or unsupported response."""


class AgentIterationLimitError(AgentError):
    """The model did not finish within the configured number of turns."""
