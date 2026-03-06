import json
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from Core.utils.ontology_utils import normalize_entity_name, normalize_entity_type


class OntologyEntityConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    entity_id: str = Field(alias="ontology_id")
    entity_name: str = Field(alias="canonical_name")
    entity_type: str
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    status: str = "active"
    ontology_source: str = "config"

    @field_validator("entity_id")
    @classmethod
    def _validate_entity_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("ontology_id cannot be empty")
        return value

    @field_validator("entity_name")
    @classmethod
    def _normalize_entity_name(cls, value: str) -> str:
        normalized = normalize_entity_name(value)
        if not normalized:
            raise ValueError("canonical_name cannot be empty")
        return normalized

    @field_validator("entity_type")
    @classmethod
    def _normalize_entity_type(cls, value: str) -> str:
        normalized = normalize_entity_type(value)
        if not normalized:
            raise ValueError("entity_type cannot be empty")
        return normalized

    @field_validator("aliases", "keywords")
    @classmethod
    def _normalize_terms(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        for value in values or []:
            item = normalize_entity_name(value)
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def _ensure_canonical_alias(self) -> "OntologyEntityConfig":
        if self.entity_name not in self.aliases:
            self.aliases.insert(0, self.entity_name)
        return self


class OntologyConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    path: Optional[str] = None
    entities: List[OntologyEntityConfig] = Field(default_factory=list)
    mapping_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_provisional_entities: bool = True
    use_query_resolution: bool = True

    @model_validator(mode="after")
    def _load_entities_from_path(self) -> "OntologyConfig":
        if not self.path:
            return self

        loaded_entities = self._read_entities_file(self.path)
        merged = list(self.entities)
        seen_ids = {entity.entity_id for entity in merged}
        for entity in loaded_entities:
            if entity.entity_id not in seen_ids:
                merged.append(entity)
                seen_ids.add(entity.entity_id)
        self.entities = merged
        return self

    @staticmethod
    def _read_entities_file(path: str) -> List[OntologyEntityConfig]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ontology file not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".json"):
                payload = json.load(handle)
            else:
                payload = yaml.safe_load(handle)

        raw_entities = payload.get("entities", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_entities, list):
            raise ValueError("Ontology file must contain a list of entities or an 'entities' list")

        return [OntologyEntityConfig(**entity) for entity in raw_entities]