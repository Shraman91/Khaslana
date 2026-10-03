"""Hybrid score fusion and file-priority boost ranker."""

from __future__ import annotations

from typing import Literal

from khaslana.retrieval.models import Chunk, SearchResult


class HybridRanker:
    """Combines dense semantic vector scores and sparse BM25 scores with priority boosts."""

    def __init__(
        self,
        default_fusion: Literal["weighted", "rrf"] = "weighted",
        default_semantic_weight: float = 0.6,
        default_bm25_weight: float = 0.4,
        focused_file_boost: float = 1.25,
        modified_file_boost: float = 1.15,
    ) -> None:
        self.default_fusion = default_fusion
        self.default_semantic_weight = default_semantic_weight
        self.default_bm25_weight = default_bm25_weight
        self.focused_file_boost = focused_file_boost
        self.modified_file_boost = modified_file_boost

    def compute_boost(
        self,
        chunk: Chunk,
        focused_file: str | None = None,
        modified_files: set[str] | None = None,
    ) -> float:
        """Calculate total multiplier boost for a chunk."""
        boost = 1.0
        if focused_file:
            # Check exact or normalized relative path match
            norm_focused = focused_file.replace("\\", "/").lstrip("./")
            norm_chunk_file = chunk.file.replace("\\", "/").lstrip("./")
            if (
                norm_chunk_file == norm_focused
                or norm_chunk_file.endswith(norm_focused)
                or norm_focused.endswith(norm_chunk_file)
            ):
                boost *= self.focused_file_boost

        if modified_files:
            norm_chunk_file = chunk.file.replace("\\", "/").lstrip("./")
            for mod_file in modified_files:
                norm_mod = mod_file.replace("\\", "/").lstrip("./")
                if norm_chunk_file == norm_mod or norm_chunk_file.endswith(norm_mod):
                    boost *= self.modified_file_boost
                    break

        return boost

    def rank(
        self,
        semantic_results: list[SearchResult],
        bm25_scores: dict[str, float],
        all_chunks_by_id: dict[str, Chunk],
        fusion: Literal["weighted", "rrf"] | None = None,
        semantic_weight: float | None = None,
        bm25_weight: float | None = None,
        focused_file: str | None = None,
        modified_files: set[str] | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Fuse and rank search results."""
        strategy = fusion or self.default_fusion
        w_sem = (
            semantic_weight
            if semantic_weight is not None
            else self.default_semantic_weight
        )
        w_bm25 = (
            bm25_weight
            if bm25_weight is not None
            else self.default_bm25_weight
        )

        semantic_map = {res.chunk.id: res.score for res in semantic_results}
        all_candidate_ids = set(semantic_map.keys()).union(set(bm25_scores.keys()))

        if not all_candidate_ids:
            return []

        scored_results: list[SearchResult] = []

        if strategy == "rrf":
            # Reciprocal Rank Fusion (k = 60 standard)
            k_rrf = 60.0

            # Rank positions (1-based)
            sorted_sem = sorted(
                semantic_map.items(), key=lambda x: x[1], reverse=True
            )
            sem_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(sorted_sem)}

            sorted_bm25 = sorted(
                bm25_scores.items(), key=lambda x: x[1], reverse=True
            )
            bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(sorted_bm25)}

            for cid in all_candidate_ids:
                chunk = all_chunks_by_id.get(cid)
                if not chunk:
                    # Lookup from semantic results
                    for sr in semantic_results:
                        if sr.chunk.id == cid:
                            chunk = sr.chunk
                            break
                if not chunk:
                    continue

                r_sem = sem_ranks.get(cid)
                r_bm25 = bm25_ranks.get(cid)

                score_sem_rrf = (1.0 / (k_rrf + r_sem)) if r_sem else 0.0
                score_bm25_rrf = (1.0 / (k_rrf + r_bm25)) if r_bm25 else 0.0

                base_score = score_sem_rrf + score_bm25_rrf
                boost = self.compute_boost(
                    chunk, focused_file=focused_file, modified_files=modified_files
                )
                final_score = base_score * boost

                scored_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(final_score, 6),
                        semantic_score=semantic_map.get(cid),
                        bm25_score=bm25_scores.get(cid),
                        boost=boost,
                        metadata={
                            "strategy": "rrf",
                            "semantic_rank": r_sem,
                            "bm25_rank": r_bm25,
                        },
                    )
                )

        else:
            # Weighted normalization fusion
            for cid in all_candidate_ids:
                chunk = all_chunks_by_id.get(cid)
                if not chunk:
                    for sr in semantic_results:
                        if sr.chunk.id == cid:
                            chunk = sr.chunk
                            break
                if not chunk:
                    continue

                sem_score = semantic_map.get(cid, 0.0)
                bm25_score = bm25_scores.get(cid, 0.0)

                base_score = (w_sem * sem_score) + (w_bm25 * bm25_score)
                boost = self.compute_boost(
                    chunk, focused_file=focused_file, modified_files=modified_files
                )
                final_score = base_score * boost

                scored_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(final_score, 4),
                        semantic_score=sem_score if cid in semantic_map else None,
                        bm25_score=bm25_score if cid in bm25_scores else None,
                        boost=boost,
                        metadata={
                            "strategy": "weighted",
                            "semantic_weight": w_sem,
                            "bm25_weight": w_bm25,
                        },
                    )
                )

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]
