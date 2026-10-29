"""
LlamaIndex VectorStore integration for HyperspaceDB.

Provides:
- High-throughput batch ingestion (`batch_insert`) with Sidecar Payload storage.
- Automatic KAG graph edge extraction from LlamaIndex `node.relationships` (`PARENT`, `NEXT`, `PREVIOUS`, `SOURCE`, `CHILD`) and `metadata["relations"]` via `link_batch`.
- Full `MetadataFilters` pushdown (`EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `TEXT_MATCH`, `IN`, `AND`, `OR`, `NOT`, plus spatial `$in_cone`, `$in_ball`, `$in_box`).
- Multi-Manifold support (`l2`, `cosine`, `dot`, `poincare`, `lorentz`, `hybrid` 801D, `manifold_product`).
- Native `hybrid_search`, `vector_graph_search` (with Schrodinger Wave Resonance), `traverse_graph`, and Tribunal Router (`verify_generation`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition as LlamaFilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from hyperspace import HyperspaceClient, VerificationResult

logger = logging.getLogger(__name__)


def _deterministic_node_id(raw_id: Any) -> int:
    """Convert any node ID into a deterministic non-zero uint32 for HyperspaceDB."""
    if isinstance(raw_id, int) and 0 < raw_id <= 0xFFFFFFFF:
        return raw_id
    s = str(raw_id)
    if s.isdigit():
        val = int(s)
        if 0 < val <= 0xFFFFFFFF:
            return val
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest()
    val = int.from_bytes(digest, "big") & 0x7FFFFFFF
    return val if val != 0 else 1


def _convert_single_filter(mf: Any) -> Optional[Dict[str, Any]]:
    """Convert a single LlamaIndex MetadataFilter or dict into a HyperspaceDB FilterCondition dict."""
    if isinstance(mf, dict):
        key = str(mf.get("key", ""))
        val = mf.get("value")
        if isinstance(val, dict):
            if "$in_ball" in val:
                ball = val["$in_ball"]
                return {"in_ball": {"center": [float(x) for x in ball["center"]], "radius": float(ball["radius"])}}
            if "$in_cone" in val:
                cone = val["$in_cone"]
                return {"in_cone": {"apex": [float(x) for x in cone["apex"]], "aperture": float(cone["aperture"])}}
            if "$in_box" in val:
                box = val["$in_box"]
                return {"in_box": {"min": [float(x) for x in box["min"]], "max": [float(x) for x in box["max"]]}}
        return {"match": {"key": key, "value": str(val)}}

    if isinstance(mf, MetadataFilters):
        return _convert_llamaindex_filters(mf)

    if not isinstance(mf, MetadataFilter):
        return None

    key = str(mf.key)
    val = mf.value
    op = mf.operator

    if isinstance(val, dict):
        if "$in_ball" in val:
            ball = val["$in_ball"]
            return {"in_ball": {"center": [float(x) for x in ball["center"]], "radius": float(ball["radius"])}}
        if "$in_cone" in val:
            cone = val["$in_cone"]
            return {"in_cone": {"apex": [float(x) for x in cone["apex"]], "aperture": float(cone["aperture"])}}
        if "$in_box" in val:
            box = val["$in_box"]
            return {"in_box": {"min": [float(x) for x in box["min"]], "max": [float(x) for x in box["max"]]}}

    if op == FilterOperator.EQ:
        return {"match": {"key": key, "value": str(val)}}
    if op == FilterOperator.NE:
        return {"not": {"match": {"key": key, "value": str(val)}}}
    if op == FilterOperator.GT:
        return {"range": {"key": key, "gt": float(val)}}
    if op == FilterOperator.GTE:
        return {"range": {"key": key, "gte": float(val)}}
    if op == FilterOperator.LT:
        return {"range": {"key": key, "lt": float(val)}}
    if op == FilterOperator.LTE:
        return {"range": {"key": key, "lte": float(val)}}
    if op == FilterOperator.TEXT_MATCH:
        return {"prefix": {"key": key, "prefix": str(val)}}
    if op == FilterOperator.IN and isinstance(val, (list, tuple, set)):
        return {"or": [{"match": {"key": key, "value": str(item)}} for item in val]}
    if op == FilterOperator.NIN and isinstance(val, (list, tuple, set)):
        return {"not": {"or": [{"match": {"key": key, "value": str(item)}} for item in val]}}

    return {"match": {"key": key, "value": str(val)}}


def _convert_llamaindex_filters(filters: Optional[MetadataFilters]) -> Optional[Dict[str, Any]]:
    """Translate LlamaIndex MetadataFilters into a HyperspaceDB FilterCondition dictionary."""
    if filters is None or not filters.filters:
        return None

    converted: List[Dict[str, Any]] = []
    for item in filters.filters:
        c = _convert_single_filter(item)
        if c is not None:
            converted.append(c)

    if not converted:
        return None
    if len(converted) == 1 and filters.condition != LlamaFilterCondition.NOT:
        return converted[0]

    cond = str(getattr(filters.condition, "value", filters.condition)).lower()
    if cond == "or":
        return {"or": converted}
    if cond == "not":
        return {"not": converted[0] if len(converted) == 1 else {"and": converted}}
    return {"and": converted}


_RELATIONSHIP_LABELS = {
    NodeRelationship.SOURCE: "DERIVED_FROM",
    NodeRelationship.PARENT: "CHILD_OF",
    NodeRelationship.CHILD: "PARENT_OF",
    NodeRelationship.PREVIOUS: "PREVIOUS_CHUNK",
    NodeRelationship.NEXT: "NEXT_CHUNK",
}


class HyperspaceVectorStore(VectorStore):
    """
    HyperspaceDB Vector Store for LlamaIndex.

    Supports Euclidean, Spherical, Hyperbolic (Poincare, Lorentz), Hybrid (33D Lorentz + 768D Cosine),
    and Manifold Product spaces, with automatic CSR KAG graph edge indexing.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        collection_name: str = "llamaindex_default",
        host: str = "localhost",
        port: int = 50051,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        encryption_key: Optional[str] = None,
        client: Optional[HyperspaceClient] = None,
        dimension: int = 1536,
        metric: str = "l2",
        quantization: str = "none",
        enable_kag_edges: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._collection_name = collection_name
        self.encryption_key = encryption_key
        self.dimension = dimension
        self.metric = metric
        self.quantization = quantization
        self.enable_kag_edges = enable_kag_edges

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            target = host if ":" in host else f"{host}:{port}"
            self._client = HyperspaceClient(
                host=target,
                api_key=api_key,
                user_id=user_id,
            )
            self._owns_client = True

        if encryption_key:
            self._client.register_collection_key(collection_name, encryption_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = self._client.list_collections()
            existing = next((c for c in collections if c.get("name") == self._collection_name), None)
            if existing:
                return
            self._client.create_collection(
                name=self._collection_name,
                dimension=self.dimension,
                metric=self.metric,
                quantization=self.quantization,
            )
        except Exception as e:
            logger.debug("Collection check/creation skipped for '%s': %s", self._collection_name, e)

    @property
    def client(self) -> HyperspaceClient:
        """Return the underlying HyperspaceClient."""
        return self._client

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Batch-add LlamaIndex nodes to HyperspaceDB and link their structural relationships
        in the in-engine CSR KAG graph.
        """
        if not nodes:
            return []

        ids: List[int] = []
        vectors: List[List[float]] = []
        metadatas: List[Dict[str, str]] = []
        payloads: List[bytes] = []
        kag_edges: List[Dict[str, Any]] = []
        node_ids_out: List[str] = []

        for node in nodes:
            hs_id = _deterministic_node_id(node.node_id)
            raw_meta = node_to_metadata_dict(node, remove_text=True, flat_metadata=True)
            str_meta: Dict[str, str] = {str(k): str(v) for k, v in raw_meta.items() if v is not None}
            str_meta["_node_id"] = str(node.node_id)
            if node.ref_doc_id:
                str_meta["ref_doc_id"] = str(node.ref_doc_id)

            embedding = node.get_embedding()
            content = node.get_content() or ""

            ids.append(hs_id)
            vectors.append([float(x) for x in embedding])
            metadatas.append(str_meta)
            payloads.append(content.encode("utf-8"))
            node_ids_out.append(str(node.node_id))

            if self.enable_kag_edges:
                # 1. Extract structural relationships from LlamaIndex node.relationships
                relationships = getattr(node, "relationships", None) or {}
                for rel_type, rel_info in relationships.items():
                    label = _RELATIONSHIP_LABELS.get(rel_type, str(rel_type))
                    targets = rel_info if isinstance(rel_info, list) else [rel_info]
                    for target in targets:
                        target_node_id = getattr(target, "node_id", None)
                        if target_node_id:
                            kag_edges.append(
                                {
                                    "source_id": hs_id,
                                    "target_id": _deterministic_node_id(target_node_id),
                                    "relation": label,
                                    "weight": 1.0,
                                }
                            )

                # 2. Extract explicit relations from node.metadata["relations"]
                custom_rels = (node.metadata or {}).get("relations")
                if isinstance(custom_rels, list):
                    for rel in custom_rels:
                        if isinstance(rel, dict) and "target_id" in rel:
                            kag_edges.append(
                                {
                                    "source_id": hs_id,
                                    "target_id": _deterministic_node_id(rel["target_id"]),
                                    "relation": str(rel.get("relation", "RELATED_TO")),
                                    "weight": float(rel.get("weight", 1.0)),
                                }
                            )

        self._client.batch_insert(
            vectors=vectors,
            ids=ids,
            collection=self._collection_name,
            metadatas=metadatas,
            payloads=payloads,
        )

        if kag_edges:
            try:
                self._client.link_batch(kag_edges, collection=self._collection_name)
            except Exception as e:
                logger.debug("Non-fatal KAG link_batch warning: %s", e)

        return node_ids_out

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete all nodes belonging to `ref_doc_id` (or direct node ID).
        """
        # 1. Try deleting all chunks matching ref_doc_id via scroll
        deleted_any = False
        try:
            points, _ = self._client.scroll(
                collection=self._collection_name,
                limit=500,
                filter={"match": {"key": "ref_doc_id", "value": str(ref_doc_id)}},
            )
            if points:
                point_ids = [int(p["id"]) for p in points if "id" in p]
                if point_ids:
                    self._client.delete_batch(point_ids, collection=self._collection_name)
                    deleted_any = True
        except Exception:
            pass

        # 2. Also delete deterministic ID directly
        if not deleted_any:
            hs_id = _deterministic_node_id(ref_doc_id)
            try:
                self._client.delete(id=hs_id, collection=self._collection_name)
            except Exception as e:
                logger.warning("Delete failed for ref_doc_id=%s: %s", ref_doc_id, e)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute a vector, hybrid, or filtered query against HyperspaceDB.
        """
        hs_filter = _convert_llamaindex_filters(query.filters)
        if "filter" in kwargs and kwargs["filter"] is not None:
            hs_filter = kwargs["filter"]

        mode = query.mode
        is_hybrid = mode in (VectorStoreQueryMode.HYBRID, "hybrid")
        alpha = (query.alpha if query.alpha is not None else 0.5) if is_hybrid else None

        results = self._client.search(
            vector=query.query_embedding or [],
            top_k=query.similarity_top_k,
            collection=self._collection_name,
            filter=hs_filter,
            hybrid_alpha=alpha,
            hybrid_query=query.query_str if is_hybrid else None,
            include_payload=True,
        )

        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for res in results:
            metadata = dict(res.get("metadata", {}) or {})
            payload = res.get("payload")
            if isinstance(payload, (bytes, bytearray)):
                text = payload.decode("utf-8", errors="replace")
            elif isinstance(payload, str):
                text = payload
            else:
                text = metadata.get("text", "")

            original_node_id = metadata.get("_node_id", str(res.get("id")))
            try:
                node = metadata_dict_to_node(metadata)
                if isinstance(node, TextNode) and text:
                    node.text = text
            except Exception:
                node = TextNode(id_=original_node_id, text=text, metadata=metadata)

            dist = float(res.get("distance", 0.0))
            score = float(res.get("score", 1.0 / (1.0 + max(0.0, dist))))

            nodes.append(node)
            similarities.append(score)
            ids.append(original_node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def hybrid_search(
        self,
        query_str: str,
        query_embedding: List[float],
        similarity_top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        """Convenience method for server-side Hybrid Dense + BM25 search."""
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_str,
            similarity_top_k=similarity_top_k,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=alpha,
            filters=filters,
        )
        return self.query(q)

    def vector_graph_search(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 5,
        max_depth: int = 2,
        decay_factor: float = 0.85,
        relation_filter: Optional[str] = None,
        use_wave: bool = False,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a 1-RPC Joint Vector-Graph Search combining dense/hyperbolic retrieval
        with multi-hop CSR KAG graph traversal and optional Wave Resonance (`use_wave=True`).
        """
        hs_filter = _convert_llamaindex_filters(filters)
        return self._client.vector_graph_search(
            vector=query_embedding,
            top_k=similarity_top_k,
            max_depth=max_depth,
            decay_factor=decay_factor,
            collection=self._collection_name,
            relation_filter=relation_filter,
            filter=hs_filter,
            use_wave=use_wave,
        )

    def traverse_graph(
        self,
        start_node_id: Any,
        max_depth: int = 2,
        decay_factor: float = 0.85,
        relation_filter: Optional[str] = None,
        use_wave: bool = False,
    ) -> List[Dict[str, Any]]:
        """Traverse explicit KAG edges from `start_node_id` up to `max_depth` hops."""
        num_id = _deterministic_node_id(start_node_id)
        return self._client.traverse(
            start_node_id=num_id,
            max_depth=max_depth,
            decay_factor=decay_factor,
            collection=self._collection_name,
            relation_filter=relation_filter,
            use_wave=use_wave,
        )

    def verify_generation(
        self,
        query_embedding: List[float],
        answer_embedding: List[float],
        context_embeddings: List[List[float]],
        context_node_ids: Optional[List[Any]] = None,
        entailment_aperture: float = 0.65,
        max_geodesic_distance: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Tribunal Router (Mathematical Anti-Hallucination Guardrail).

        Evaluates whether an LLM response (`answer_embedding`) is geometrically entailed
        by the retrieved LlamaIndex context nodes in hyperbolic/hybrid manifold space.
        """
        numeric_ids = [_deterministic_node_id(cid) for cid in (context_node_ids or [])]
        result: VerificationResult = self._client.verify_generation(
            query_vec=query_embedding,
            answer_vec=answer_embedding,
            context_vecs=context_embeddings,
            context_node_ids=numeric_ids,
            collection=self._collection_name,
            entailment_aperture=entailment_aperture,
            max_geodesic_distance=max_geodesic_distance,
        )
        return {
            "verified": result.verified,
            "confidence": result.confidence,
            "entailment_score": result.entailment_score,
            "min_geodesic_distance": result.min_geodesic_distance,
            "supporting_node_ids": result.supporting_node_ids,
            "contradicting_node_ids": result.contradicting_node_ids,
            "correction_prompt": result.correction_prompt,
        }

    def get_cache_stats(self) -> dict:
        """Get cache statistics for L0 Hot Tier Cache of the collection."""
        return self._client.get_cache_stats(self._collection_name)

    def clear_cache(self) -> bool:
        """Purge L0 Cache items for the collection."""
        return self._client.clear_cache(self._collection_name)

    def update_cache_config(self, policy: str, ann_threshold: Optional[float] = None) -> bool:
        """Update L0 Cache configuration for the collection."""
        return self._client.update_cache_config(self._collection_name, policy, ann_threshold)

    def link(self, source_id: Any, target_id: Any, relation: str = "OWNS", weight: float = 1.0) -> bool:
        """Create an explicit directed edge between nodes in the in-engine vector-graph."""
        return bool(
            self._client.link(
                _deterministic_node_id(source_id),
                _deterministic_node_id(target_id),
                relation=relation,
                weight=weight,
                collection=self._collection_name,
            )
        )

    def link_batch(self, edges: List[Dict[str, Any]]) -> int:
        """Batch-insert explicit directed edges in CSR graph."""
        normalized = [
            {
                "source_id": _deterministic_node_id(e["source_id"]),
                "target_id": _deterministic_node_id(e["target_id"]),
                "relation": str(e.get("relation", "OWNS")),
                "weight": float(e.get("weight", 1.0)),
            }
            for e in edges
        ]
        return int(self._client.link_batch(normalized, collection=self._collection_name))

    def unlink(self, source_id: Any, target_id: Any, relation: str = "OWNS") -> bool:
        """Delete an explicit directed edge with zero-lag invalidation."""
        return bool(
            self._client.unlink(
                _deterministic_node_id(source_id),
                _deterministic_node_id(target_id),
                relation=relation,
                collection=self._collection_name,
            )
        )

    def induce_edges_from_metadata(self, rules: List[Dict[str, Any]]) -> int:
        """Induce topological edges from metadata properties across document entities."""
        return int(self._client.induce_edges_from_metadata(rules, collection=self._collection_name))

    def close(self) -> None:
        """Close the underlying gRPC client if owned by this store."""
        if getattr(self, "_owns_client", False) and hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()
