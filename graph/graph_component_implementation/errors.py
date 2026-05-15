class GraphStoreError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class NodeNotFoundError(GraphStoreError):
    def __init__(self, message: str = "Node not found") -> None:
        super().__init__("E001", message)


class EdgeNotFoundError(GraphStoreError):
    def __init__(self, message: str = "Edge not found") -> None:
        super().__init__("E002", message)


class DuplicateNodeError(GraphStoreError):
    def __init__(self, message: str = "Duplicate node") -> None:
        super().__init__("E003", message)


class InvalidEmbeddingDimensionError(GraphStoreError):
    def __init__(self, message: str = "Invalid embedding dimension") -> None:
        super().__init__("E004", message)


class SerializationFailedError(GraphStoreError):
    def __init__(self, message: str = "Serialization failed") -> None:
        super().__init__("E005", message)


class ShardCorruptedError(GraphStoreError):
    def __init__(self, message: str = "Shard corrupted") -> None:
        super().__init__("E006", message)
