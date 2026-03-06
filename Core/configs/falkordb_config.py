from dataclasses import dataclass, field


@dataclass
class FalkorDBConfig:
    """Configuration for FalkorDB graph database connection."""

    host: str = "localhost"
    port: int = 6379
    username: str = ""
    password: str = ""
    graph_prefix: str = "bookrag"

    def graph_name_for_doc(self, tenant_id: str, doc_id: str) -> str:
        """Return the FalkorDB graph name for a per-document KG."""
        return f"{self.graph_prefix}:{tenant_id}:doc:{doc_id}"

    def graph_name_for_global(self, tenant_id: str) -> str:
        """Return the FalkorDB graph name for a tenant-level global KG."""
        return f"{self.graph_prefix}:{tenant_id}:global"

