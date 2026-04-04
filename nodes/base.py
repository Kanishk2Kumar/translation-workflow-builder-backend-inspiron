from abc import ABC, abstractmethod
import time


class BaseNode(ABC):
    """
    Every node receives a context dict and returns an updated context dict.
    Nodes should never mutate the incoming dict — always return a new one.
    """

    def __init__(self, node_id: str, config: dict):
        self.node_id = node_id
        self.config = config

    async def execute(self, context: dict) -> dict:
        start = time.monotonic()
        try:
            result = await self.run(context)
            ms = round((time.monotonic() - start) * 1000)
            result.setdefault("_logs", []).append({
                "node_id": self.node_id,
                "node_type": self.__class__.__name__,
                "status": "success",
                "ms": ms,
            })
            return result
        except Exception as e:
            ms = round((time.monotonic() - start) * 1000)
            context.setdefault("_logs", []).append({
                "node_id": self.node_id,
                "node_type": self.__class__.__name__,
                "status": "error",
                "error": str(e),
                "ms": ms,
            })
            raise

    @abstractmethod
    async def run(self, context: dict) -> dict:
        """
        Implement node logic here.
        Receive context, return updated context.
        """
        ...