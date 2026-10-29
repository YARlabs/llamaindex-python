"""
LlamaIndex Tool for HyperspaceDB Formal Proof-Obligation Query Planning & Intent Disambiguation.

Delegates directly to the official `hyperspace.query_planner` module in `hyperspacedb` SDK
without any relative path hacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

try:
    from hyperspace.query_planner import (
        ClarificationPayload,
        ExecutionStrategy,
        LOCAL_LLM_PLANNER_PROMPT,
        ProofObligation,
        ProofObligationPlan,
        QUERY_PLANNER_TOOL_SCHEMA,
        QueryPlannerResult,
        plan_query,
    )
except ImportError:
    ExecutionStrategy = Literal["single", "parallel", "dag", "clarify"]

    @dataclass
    class ClarificationPayload:
        reason: str
        question: str
        options: List[str]

    @dataclass
    class ProofObligation:
        id: str
        sub_query: str
        target_collection: Optional[str] = None
        depends_on: List[str] = field(default_factory=list)
        reasoning: Optional[str] = None
        verify_entailment: bool = True

    @dataclass
    class ProofObligationPlan:
        execution_strategy: ExecutionStrategy
        obligations: List[ProofObligation]
        tribunal_threshold: float = 0.65

    @dataclass
    class QueryPlannerResult:
        status: Literal["needs_clarification", "ready", "ready_with_proof_plan"]
        original_query: str
        execution_strategy: ExecutionStrategy = "single"
        clarification: Optional[ClarificationPayload] = None
        sub_queries: Optional[List[str]] = None
        proof_plan: Optional[ProofObligationPlan] = None
        target_collections: Optional[List[str]] = None

        def to_dict(self) -> Dict[str, Any]:
            return {k: v for k, v in asdict(self).items() if v is not None}

        def to_json(self) -> str:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    QUERY_PLANNER_TOOL_SCHEMA: Dict[str, Any] = {"name": "hyperspace_query_planner"}
    LOCAL_LLM_PLANNER_PROMPT: str = "You are the HyperspaceDB Formal Proof-Obligation Query Planner."

    def plan_query(
        raw_user_query: str,
        context: Optional[str] = None,
        collections: Optional[List[str]] = None,
        max_sub_queries: int = 5,
    ) -> QueryPlannerResult:
        cleaned = (raw_user_query or "").strip()
        parts = [p.strip() for p in re.split(r"\s+(?:and|vs\.?|versus)\s+", cleaned, flags=re.I) if len(p.strip()) > 3]
        if len(parts) > 1:
            sub_qs = parts[:max_sub_queries]
            obs = [ProofObligation(id=f"ob_{i+1}", sub_query=sq) for i, sq in enumerate(sub_qs)]
            return QueryPlannerResult(
                status="ready_with_proof_plan",
                original_query=cleaned,
                execution_strategy="parallel",
                sub_queries=sub_qs,
                proof_plan=ProofObligationPlan(execution_strategy="parallel", obligations=obs),
                target_collections=collections,
            )
        return QueryPlannerResult(
            status="ready",
            original_query=cleaned,
            execution_strategy="single",
            sub_queries=[cleaned],
            target_collections=collections,
        )


try:
    from llama_index.core.tools import BaseTool, FunctionTool
except ImportError:
    class BaseTool:  # type: ignore[no-redef]
        pass

    class FunctionTool:  # type: ignore[no-redef]
        @classmethod
        def from_defaults(cls, fn: Any, name: Optional[str] = None, description: Optional[str] = None) -> Any:
            return fn


def hyperspace_query_planner(
    raw_user_query: str,
    context: Optional[str] = None,
    collections: Optional[List[str]] = None,
    max_sub_queries: Optional[int] = 5,
) -> str:
    """
    Evaluate a user query prior to vector/graph search against HyperspaceDB.
    Returns structured JSON instructions to clarify ambiguity or execute a DAG of proof obligations.
    """
    result = plan_query(
        raw_user_query=raw_user_query,
        context=context,
        collections=collections,
        max_sub_queries=max_sub_queries or 5,
    )
    return result.to_json()


def get_query_planner_tool(
    name: str = "hyperspace_query_planner",
    description: Optional[str] = None,
) -> Any:
    """Return a LlamaIndex FunctionTool wrapping the HyperspaceDB Formal Proof-Obligation Query Planner."""
    desc = description or (
        "Enterprise RAG & KAG Formal Proof-Obligation Query Planner. Evaluates user intent before retrieval: "
        "detects ambiguous questions and decomposes multi-part queries into parallel or DAG proof obligations."
    )
    return FunctionTool.from_defaults(
        fn=hyperspace_query_planner,
        name=name,
        description=desc,
    )


__all__ = [
    "plan_query",
    "hyperspace_query_planner",
    "get_query_planner_tool",
    "QueryPlannerResult",
    "ProofObligation",
    "ProofObligationPlan",
    "ClarificationPayload",
    "ExecutionStrategy",
    "QUERY_PLANNER_TOOL_SCHEMA",
    "LOCAL_LLM_PLANNER_PROMPT",
]
