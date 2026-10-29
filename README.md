# LlamaIndex HyperspaceDB Integration (VectorStore & PropertyGraphStore)

[![PyPI Version](https://img.shields.io/pypi/v/llama-index-vector-stores-hyperspace?style=for-the-badge)](https://pypi.org/project/llama-index-vector-stores-hyperspace/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

Official LlamaIndex integration for **[HyperspaceDB](https://yar.ink)** — the Multi-Manifold Vector & Knowledge-Augmented Graph (KAG) Engine.

## Key Capabilities (v4.2)

- **`HyperspaceVectorStore`**: High-throughput batch ingestion (`batch_insert`), automatic CSR KAG edge linking from LlamaIndex `node.relationships` (`PARENT`, `NEXT`, `PREVIOUS`, `SOURCE`, `CHILD`), full `MetadataFilters` pushdown (including hyperbolic `$in_cone` and spatial `$in_ball`), `hybrid_search`, and `vector_graph_search` (with Wave Resonance).
- **`HyperspacePropertyGraphStore`**: Native LlamaIndex `PropertyGraphStore` implementation backed by HyperspaceDB's lock-free CSR KAG engine (`upsert_nodes`, `upsert_relations` via `link_batch`, `get_triplets`, `get_rel_map` via `traverse`, `vector_query`).
- **Tribunal Router (`verify_generation`)**: Mathematical anti-hallucination verification evaluating whether an LLM draft is entailed by retrieved LlamaIndex context nodes on the manifold.
- **Formal Proof-Obligation Query Planner**: Pre-retrieval tool (`get_query_planner_tool`) that disambiguates vague prompts and decomposes complex questions into parallel/DAG proof obligations.

## Installation

```bash
pip install llama-index-vector-stores-hyperspace
```

## Quick Start

### 1. Vector Store (`HyperspaceVectorStore`)

```python
from llama_index_hyperspace import HyperspaceVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

vector_store = HyperspaceVectorStore(
    collection_name="agent_memory",
    host="localhost",
    port=50051,
    dimension=801,
    metric="hybrid",  # 33D Lorentz Hierarchy + 768D Cosine Semantic Tail
)

# All node.relationships (PARENT, NEXT, PREVIOUS, SOURCE) are automatically indexed as KAG edges!
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

### 2. Knowledge-Augmented Property Graph (`HyperspacePropertyGraphStore`)

```python
from llama_index_hyperspace import HyperspacePropertyGraphStore
from llama_index_hyperspace.property_graph import EntityNode, Relation

graph_store = HyperspacePropertyGraphStore(
    collection_name="enterprise_kag",
    host="localhost",
    port=50051,
    use_wave=True,  # Enable Schrodinger Wave Resonance during multi-hop traversal
)

graph_store.upsert_nodes([
    EntityNode(name="MasterAgreement", label="Contract", properties={"jurisdiction": "DE"}),
    EntityNode(name="IndemnityClause", label="Clause", properties={"cap": "12_months"}),
])

graph_store.upsert_relations([
    Relation(label="CONTAINS", source_id="MasterAgreement", target_id="IndemnityClause", properties={"weight": 1.0}),
])

triplets = graph_store.get_rel_map([EntityNode(name="MasterAgreement", label="Contract")], depth=2)
```

### 3. Tribunal Router: Anti-Hallucination Guardrail

```python
verdict = vector_store.verify_generation(
    query_embedding=query_vec,
    answer_embedding=draft_answer_vec,
    context_embeddings=[node.get_embedding() for node in retrieved_nodes],
    context_node_ids=[node.node_id for node in retrieved_nodes],
    entailment_aperture=0.65,
)

if not verdict["verified"]:
    print("Hallucination rejected:", verdict["correction_prompt"])
```

## License

Apache-2.0. Copyright © 2026 **YARlabs**.
