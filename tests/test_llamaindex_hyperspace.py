"""Unit tests for LlamaIndex HyperspaceDB VectorStore and PropertyGraphStore."""

from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index_hyperspace import (
    HyperspacePropertyGraphStore,
    HyperspaceVectorStore,
    plan_query,
)


class TestLlamaIndexHyperspace:
    @patch("llama_index_hyperspace.vector_stores.HyperspaceClient")
    def test_batch_add_and_relationship_linking(self, mock_client_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.list_collections.return_value = [{"name": "test_col"}]
        mock_client_cls.return_value = mock_instance

        store = HyperspaceVectorStore(collection_name="test_col", dimension=4, metric="l2")

        n1 = TextNode(
            id_="node_1",
            text="Parent architecture overview",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"topic": "arch"},
        )
        n2 = TextNode(
            id_="node_2",
            text="Child module specification",
            embedding=[0.4, 0.3, 0.2, 0.1],
            metadata={"topic": "module"},
            relationships={NodeRelationship.PARENT: RelatedNodeInfo(node_id="node_1")},
        )

        ids = store.add([n1, n2])
        assert ids == ["node_1", "node_2"]
        mock_instance.batch_insert.assert_called_once()
        mock_instance.link_batch.assert_called_once()

    @patch("llama_index_hyperspace.vector_stores.HyperspaceClient")
    def test_metadata_filters_pushdown_and_hybrid_query(self, mock_client_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.list_collections.return_value = [{"name": "test_col"}]
        mock_instance.search.return_value = [
            {
                "id": 101,
                "distance": 0.15,
                "score": 0.87,
                "payload": b"Filtered content",
                "metadata": {"_node_id": "node_101", "category": "legal"},
            }
        ]
        mock_client_cls.return_value = mock_instance

        store = HyperspaceVectorStore(collection_name="test_col", dimension=4, metric="cosine")
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="legal", operator=FilterOperator.EQ),
                MetadataFilter(key="version", value=2, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND,
        )
        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            query_str="liability clause",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.7,
            filters=filters,
        )
        res = store.query(q)
        assert len(res.nodes) == 1
        assert res.ids == ["node_101"]
        assert res.nodes[0].get_content() == "Filtered content"

        call_kwargs = mock_instance.search.call_args.kwargs
        assert call_kwargs["hybrid_alpha"] == 0.7
        assert call_kwargs["hybrid_query"] == "liability clause"
        assert "and" in call_kwargs["filter"]

    @patch("llama_index_hyperspace.property_graph.HyperspaceClient")
    def test_property_graph_store_upsert_and_traverse(self, mock_client_cls: MagicMock) -> None:
        from llama_index_hyperspace.property_graph import EntityNode, Relation

        mock_instance = MagicMock()
        mock_instance.list_collections.return_value = [{"name": "kag_col"}]
        mock_instance.traverse.return_value = [
            {"node_id": 222, "relation": "DEPENDS_ON", "score": 0.92, "depth": 1, "metadata": {"_name": "RustCore"}}
        ]
        mock_client_cls.return_value = mock_instance

        pg = HyperspacePropertyGraphStore(collection_name="kag_col", use_wave=True)
        e1 = EntityNode(name="HyperspaceDB", label="Database", properties={"tier": "core"})
        e2 = EntityNode(name="RustCore", label="Engine", properties={"lang": "rust"})
        pg.upsert_nodes([e1, e2])
        mock_instance.batch_insert.assert_called_once()

        r1 = Relation(label="DEPENDS_ON", source_id="HyperspaceDB", target_id="RustCore", properties={"weight": 0.95})
        pg.upsert_relations([r1])
        mock_instance.link_batch.assert_called_once()

        triplets = pg.get_triplets(relation_names=["DEPENDS_ON"])
        assert len(triplets) == 1

        rel_map = pg.get_rel_map([e1], depth=2)
        assert len(rel_map) == 1
        mock_instance.traverse.assert_called_once()

    def test_query_planner_plan_query(self) -> None:
        plan = plan_query("What is the latency of Poincaré search and how does hybrid 801D work?")
        assert plan.status == "ready_for_search"
        assert len(plan.search_queries) >= 2
        assert len(plan.obligations) >= 2
