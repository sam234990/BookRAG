from dataclasses import dataclass


@dataclass
class MongoDBConfig:
    """Configuration for MongoDB connection."""

    uri: str = "mongodb://localhost:27017"
    db_prefix: str = "bookrag"
    system_db: str = "bookrag_system"

    def tenant_db_name(self, tenant_id: str) -> str:
        """Return the MongoDB database name for a given tenant."""
        return f"{self.db_prefix}_{tenant_id}"

