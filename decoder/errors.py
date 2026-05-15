class DecoderError(Exception):
    pass


class ConfigurationError(DecoderError):
    pass


class ValidationError(DecoderError):
    pass
