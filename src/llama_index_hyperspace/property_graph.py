"""
LlamaIndex PropertyGraphStore integration for HyperspaceDB's Knowledge-Augmented Graph (KAG).

Implements `HyperspacePropertyGraphStore` backed by HyperspaceDB's lock-free CSR graph engine
and multi-manifold vector store:
- `upsert_nodes`: Batch-inserts entity/chunk nodes with 801D Hybrid or custom embeddings.
- `upsert_relations`: Batch-links directed labeled edges via `link_batch`.
- `get`: Retrieves graph nodes by ID or property filters.
- `get_triplets`: Returns `(source_node, relation, target_node)` triplets matching entity names, relations, or IDs.
- `get_rel_map`: Executes multi-hop CSR graph traversal (`traverse` with optional Wave Resonance `use_wave=True`).
- `vector_query`: Executes joint 1-RPC `vector_graph_search`.
- `delete`: Removes nodes and incident edges.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hyperspace import HyperspaceClient
from llama_index_hyperspace.vector_stores import _convert_llamaindex_filters, _deterministic_node_id

logger = logging.getLogger(__name__)

try:
    from llama_index.core.graph_stores.types import (
        EntityNode,
        LabelledNode,
        PropertyGraphStore,
        Relation,
        Triplet,
    )
except ImportError:
    class PropertyGraphStore:  # type: ignore[no-redef]
        pass

    class LabelledNode:  # type: ignore[no-redef]
        def __init__(self, id: str = "", label: str = "entity", properties: Optional[Dict[str, Any]] = None) -> None:
            self.id = id
            self.label = label
            self.properties = properties or {}

    class EntityNode(LabelledNode):  # type: ignore[no-redef]
        def __init__(
            self,
            name: str = "",
            label: str = "entity",
            properties: Optional[Dict[str, Any]] = None,
            embedding: Optional[List[float]] = None,
        ) -> None:
            super().__init__(id=name, label=label, properties=properties)
            self.name = name
            self.embedding = embedding

    class Relation:  # type: ignore[no-redef]
        def __init__(
            self,
            label: str,
            source_id: str,
            target_id: str,
            properties: Optional[Dict[str, Any]] = None,
        ) -> None:
            self.label = label
            self.source_id = source_id
            self.target_id = target_id
            self.properties = properties or {}

    Triplet = Tuple[Any, Any, Any]  # type: ignore[misc]


def _default_hybrid_801d_embedding(text: str, depth_hint: int = 1) -> List[float]:
    """Generate a deterministic 801D Hybrid vector (33D Lorentz + 768D unit tail) when no embedding is provided."""
    import math

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
    radial = min(2.5, 0.25 * max(1, depth_hint))
    spatial_32: List[float] = []
    for i in range(32):
        b = digest[i % len(digest)]
        spatial_32.append(((b / 127.5) - 1.0) * (radial / math.sqrt(32.0)))
    norm_sq = sum(x * x for x in spatial_32)
    x0 = math.sqrt(1.0 + norm_sq)
    lorentz_33 = [x0] + spatial_32

    tail_768: List[float] = []
    for j in range(768):
        b = digest[(j + 17) % len(digest)]
        tail_768.append((b / 127.5) - 1.0)
    tail_norm = math.sqrt(sum(x * x for x in tail_768)) or 1.0
    tail_768 = [x / tail_norm for x in tail_768]
    return lorentz_33 + tail_768


class HyperspacePropertyGraphStore(PropertyGraphStore):
    """
    HyperspaceDB Knowledge-Augmented Graph (KAG) PropertyGraphStore for LlamaIndex.

    Stores both property graph entities/chunks and explicit CSR directed edges inside a single
    unified HyperspaceDB collection, eliminating the need for a separate graph database (e.g. Neo4j).
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = True

    def __init__(
        self,
        collection_name: str = "llamaindex_kag_graph",
        host: str = "localhost",
        port: int = 50051,
        api_key: Optional[str] = None,
        client: Optional[HyperspaceClient] = None,
        dimension: int = 801,
        metric: str = "hybrid",
        use_wave: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric = metric
        self.use_wave = use_wave

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            target = host if ":" in host else f"{host}:{port}"
            self._client = HyperspaceClient(host=target, api_key=api_key)
            self._owns_client = True

        # Local cache of node metadata & relations for instant triplet reconstruction
        self._node_cache: Dict[int, Any] = {}
        self._relations_cache: List[Any] = []
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            cols = self._client.list_collections()
            if not any(c.get("name") == self.collection_name for c in cols):
                self._client.create_collection(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                )
        except Exception as e:
            logger.debug("PropertyGraphStore collection init skipped: %s", e)

    @property
    def client(self) -> HyperspaceClient:
        """Return underlying HyperspaceClient."""
        return self._client

    def upsert_nodes(self, nodes: Sequence[Any]) -> None:
        """Batch-upsert EntityNode / ChunkNode instances into HyperspaceDB."""
        if not nodes:
            return

        ids: List[int] = []
        vectors: List[List[float]] = []
        metadatas: List[Dict[str, str]] = []
        payloads: List[bytes] = []

        for node in nodes:
            node_id_str = str(getattr(node, "id", None) or getattr(node, "name", ""))
            num_id = _deterministic_node_id(node_id_str)
            label = str(getattr(node, "label", "entity"))
            name = str(getattr(node, "name", node_id_str))
            props = dict(getattr(node, "properties", {}) or {})
            emb = getattr(node, "embedding", None)

            if not emb or len(emb) != self.dimension:
                if self.dimension == 801:
                    emb = _default_hybrid_801d_embedding(f"{label}:{name}:{json.dumps(props, default=str)}")
                else:
                    emb = [0.01] * self.dimension

            meta: Dict[str, str] = {str(k): str(v) for k, v in props.items() if v is not None}
            meta["_node_id"] = node_id_str
            meta["_label"] = label
            meta["_name"] = name

            text = str(getattr(node, "text", None) or name)
            ids.append(num_id)
            vectors.append([float(x) for x in emb])
            metadatas.append(meta)
            payloads.append(text.encode("utf-8"))
            self._node_cache[num_id] = node

        self._client.batch_insert(
            vectors=vectors,
            ids=ids,
            collection=self.collection_name,
            metadatas=metadatas,
            payloads=payloads,
        )

    def upsert_relations(self, relations: Sequence[Any]) -> None:
        """Batch-upsert directed labeled relations into HyperspaceDB's CSR KAG engine."""
        if not relations:
            return

        edges: List[Dict[str, Any]] = []
        for rel in relations:
            src_id = _deterministic_node_id(getattr(rel, "source_id", ""))
            tgt_id = _deterministic_node_id(getattr(rel, "target_id", ""))
            label = str(getattr(rel, "label", "RELATED_TO"))
            props = getattr(rel, "properties", {}) or {}
            weight = float(props.get("weight", 1.0))
            edges.append(
                {
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "relation": label,
                    "weight": weight,
                }
            )
            self._relations_cache.append(rel)

        self._client.link_batch(edges, collection=self.collection_name)

    def get(
        self,
        properties: Optional[Dict[str, Any]] = None,
        ids: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Retrieve LabelledNodes by explicit IDs or metadata property filters."""
        if ids:
            num_ids = [_deterministic_node_id(i) for i in ids]
            out: List[Any] = []
            try:
                pts = self._client.get_points(num_ids, collection=self.collection_name)
                for pt in pts:
                    nid = int(pt.get("id", 0))
                    if nid in self._node_cache:
                        out.append(self._node_cache[nid])
                    else:
                        meta = pt.get("metadata", {}) or {}
                        out.append(
                            EntityNode(
                                name=meta.get("_name", str(nid)),
                                label=meta.get("_label", "entity"),
                                properties=meta,
                                embedding=pt.get("vector"),
                            )
                        )
                return out
            except Exception:
                return [self._node_cache[nid] for nid in num_ids if nid in self._node_cache]

        filter_cond: Optional[Dict[str, Any]] = None
        if properties:
            clauses = [{"match": {"key": str(k), "value": str(v)}} for k, v in properties.items()]
            filter_cond = clauses[0] if len(clauses) == 1 else {"and": clauses}

        try:
            pts, _ = self._client.scroll(
                collection=self.collection_name,
                limit=200,
                filter=filter_cond,
            )
            nodes: List[Any] = []
            for pt in pts:
                nid = int(pt.get("id", 0))
                if nid in self._node_cache:
                    nodes.append(self._node_cache[nid])
                else:
                    meta = pt.get("metadata", {}) or {}
                    nodes.append(
                        EntityNode(
                            name=meta.get("_name", str(nid)),
                            label=meta.get("_label", "entity"),
                            properties=meta,
                        )
                    )
            return nodes
        except Exception:
            return list(self._node_cache.values())

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """Return `(source_node, relation, target_node)` triplets matching filters."""
        triplets: List[Any] = []
        entity_set = set(entity_names) if entity_names else None
        rel_set = set(relation_names) if relation_names else None
        id_set = set(str(i) for i in ids) if ids else None

        for rel in self._relations_cache:
            src_str = str(getattr(rel, "source_id", ""))
            tgt_str = str(getattr(rel, "target_id", ""))
            label = str(getattr(rel, "label", "RELATED_TO"))

            if rel_set and label not in rel_set:
                continue
            if id_set and src_str not in id_set and tgt_str not in id_set:
                continue

            src_num = _deterministic_node_id(src_str)
            tgt_num = _deterministic_node_id(tgt_str)
            src_node = self._node_cache.get(src_num) or EntityNode(name=src_str, label="entity")
            tgt_node = self._node_cache.get(tgt_num) or EntityNode(name=tgt_str, label="entity")

            src_name = getattr(src_node, "name", src_str)
            tgt_name = getattr(tgt_node, "name", tgt_str)
            if entity_set and src_name not in entity_set and tgt_name not in entity_set:
                continue

            triplets.append((src_node, rel, tgt_node))
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[Any],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Execute multi-hop CSR graph traversal starting from `graph_nodes` via `client.traverse()`.
        """
        ignore_set = set(ignore_rels or [])
        discovered_triplets: List[Any] = []

        for start_node in graph_nodes:
            start_id_str = str(getattr(start_node, "id", None) or getattr(start_node, "name", ""))
            start_num = _deterministic_node_id(start_id_str)
            try:
                hops = self._client.traverse(
                    start_node_id=start_num,
                    max_depth=depth,
                    decay_factor=0.85,
                    collection=self.collection_name,
                    use_wave=self.use_wave,
                )
                for hop in hops[:limit]:
                    target_num = int(hop.get("node_id", hop.get("id", 0)))
                    rel_label = str(hop.get("relation", "CONNECTED_TO"))
                    if rel_label in ignore_set or target_num == start_num:
                        continue
                    tgt_node = self._node_cache.get(target_num) or EntityNode(
                        name=str(target_num),
                        label="entity",
                        properties=hop.get("metadata", {}) or {},
                    )
                    rel = Relation(
                        label=rel_label,
                        source_id=start_id_str,
                        target_id=str(getattr(tgt_node, "id", target_num)),
                        properties={"score": hop.get("score", 1.0), "depth": hop.get("depth", 1)},
                    )
                    discovered_triplets.append((start_node, rel, tgt_node))
            except Exception as e:
                logger.debug("Traverse warning for node %s: %s", start_id_str, e)

        return discovered_triplets[:limit]

    def vector_query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> Tuple[List[Any], List[float]]:
        """Execute a joint vector-graph query over entity nodes in HyperspaceDB."""
        hs_filter = _convert_llamaindex_filters(query.filters)
        hits = self._client.vector_graph_search(
            vector=query.query_embedding or [],
            top_k=query.similarity_top_k,
            max_depth=int(kwargs.get("max_depth", 2)),
            decay_factor=float(kwargs.get("decay_factor", 0.85)),
            collection=self.collection_name,
            filter=hs_filter,
            use_wave=bool(kwargs.get("use_wave", self.use_wave)),
        )
        nodes: List[Any] = []
        scores: List[float] = []
        for hit in hits:
            nid = int(hit.get("id", hit.get("node_id", 0)))
            meta = hit.get("metadata", {}) or {}
            node = self._node_cache.get(nid) or EntityNode(
                name=meta.get("_name", str(nid)),
                label=meta.get("_label", "entity"),
                properties=meta,
            )
            nodes.append(node)
            scores.append(float(hit.get("score", 1.0)))
        return nodes, scores

    def structured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a structured property/triplet query against the KAG store.
        Supports JSON filter dicts or relation/entity name lookups.
        """
        params = param_map or {}
        if "relation_names" in params or "entity_names" in params:
            return self.get_triplets(
                entity_names=params.get("entity_names"),
                relation_names=params.get("relation_names"),
                properties=params.get("properties"),
                ids=params.get("ids"),
            )
        try:
            parsed = json.loads(query)
            if isinstance(parsed, dict):
                return self.get(properties=parsed.get("properties"), ids=parsed.get("ids"))
        except Exception:
            pass
        return self.get_triplets(entity_names=[query] if query else None)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching nodes and/or relations from the property graph."""
        target_ids: List[int] = []
        if ids:
            target_ids.extend(_deterministic_node_id(i) for i in ids)
        if entity_names:
            target_ids.extend(_deterministic_node_id(name) for name in entity_names)

        if target_ids:
            try:
                self._client.delete_batch(target_ids, collection=self.collection_name)
            except Exception:
                for tid in target_ids:
                    try:
                        self._client.delete(tid, collection=self.collection_name)
                    except Exception:
                        pass
            for tid in target_ids:
                self._node_cache.pop(tid, None)

        if relation_names:
            rel_set = set(relation_names)
            remaining = []
            for rel in self._relations_cache:
                if getattr(rel, "label", "") in rel_set:
                    try:
                        self._client.unlink(
                            _deterministic_node_id(rel.source_id),
                            _deterministic_node_id(rel.target_id),
                            relation=rel.label,
                            collection=self.collection_name,
                        )
                    except Exception:
                        pass
                else:
                    remaining.append(rel)
            self._relations_cache = remaining

    def induce_edges_from_metadata(self, rules: List[Dict[str, Any]]) -> int:
        """Declaratively induce KAG edges across nodes from metadata rules."""
        return int(self._client.induce_edges_from_metadata(rules, collection=self.collection_name))
