import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from .config import KGBuilderConfig
from .document_processor import DocumentProcessor
from .entity_extractor import EntityExtractor
from .triple_extractor import TripleExtractor
from .relation_mapper import RelationMapper
from .entity_resolver import EntityResolver
from .confidence_scorer import ConfidenceScorer
from .graph_builder import GraphBuilder
from .active_learner import ActiveLearner

logger = logging.getLogger(__name__)


class KGBuilderPipeline:
    def __init__(self, config: Optional[KGBuilderConfig] = None):
        self.config = config or KGBuilderConfig()
        self._initialized = False
        self.doc_processor: Optional[DocumentProcessor] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        self.triple_extractor: Optional[TripleExtractor] = None
        self.relation_mapper: Optional[RelationMapper] = None
        self.entity_resolver: Optional[EntityResolver] = None
        self.confidence_scorer: Optional[ConfidenceScorer] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.active_learner: Optional[ActiveLearner] = None
        self._sbert: Optional[SentenceTransformer] = None

    def initialize(self):
        if self._initialized:
            return
        logger.info("Initializing KGBuilder pipeline...")
        logger.info("spaCy model: %s", self.config.spaCy_model)
        logger.info("SBERT model: %s", self.config.sbert_model)
        logger.info("Cascade levels: %s", self.config.cascade_levels)

        self.doc_processor = DocumentProcessor(model_name=self.config.spaCy_model)
        self.entity_extractor = EntityExtractor(
            ner_labels=self.config.ner_labels,
            sbert_model_name=self.config.sbert_model,
            min_np_length=self.config.min_np_length,
        )
        self.triple_extractor = TripleExtractor(
            enable_llm=self.config.enable_spacy_llm,
            llm_model=self.config.llm_model,
        )
        self.relation_mapper = RelationMapper(
            relation_map=self.config.relation_map,
            antonym_triggers=self.config.antonym_triggers,
            synonym_triggers=self.config.synonym_triggers,
            sbert_model_name=self.config.sbert_model,
        )
        self.entity_resolver = EntityResolver(
            embed_similarity_threshold=self.config.embed_similarity_threshold,
            graph_neighbor_overlap_threshold=self.config.graph_neighbor_overlap_threshold,
            sbert_model_name=self.config.sbert_model,
        )
        self.confidence_scorer = ConfidenceScorer(
            pattern_weight=self.config.pattern_weight,
            resolution_weight=self.config.resolution_weight,
            frequency_weight=self.config.frequency_weight,
            min_confidence=self.config.min_confidence,
        )
        self.graph_builder = GraphBuilder(
            resolver=self.entity_resolver,
            scorer=self.confidence_scorer,
            min_confidence=self.config.min_confidence,
        )
        self.active_learner = ActiveLearner()
        self._sbert = SentenceTransformer(self.config.sbert_model)
        self._initialized = True
        logger.info("KGBuilder pipeline initialized")

    def process_corpus(
        self, texts: List[str], show_progress: bool = True,
        max_docs: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.initialize()
        if max_docs is not None:
            texts = texts[:max_docs]
        total = len(texts)
        all_triple_results: List[Dict] = []
        logger.info("Processing %d documents...", total)
        t_start = time.time()
        for doc_idx, text in enumerate(texts):
            if show_progress and (doc_idx + 1) % max(1, total // 10) == 0:
                elapsed = time.time() - t_start
                rate = (doc_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info("[%d/%d] %.0f docs/sec, %d triples so far",
                            doc_idx + 1, total, rate, len(all_triple_results))
            try:
                sentences = self.doc_processor.process(text)
            except Exception as e:
                logger.debug("Document %d processing failed: %s", doc_idx, e)
                continue
            for sent in sentences:
                if len(sent.get("entities", [])) == 0 and len(sent.get("text", "")) < 10:
                    continue
                extracted = self.triple_extractor.extract_or_escalate(sent)
                entities = self.entity_extractor.extract(sent)
                sentence_entity_set = set(entities)
                for item in extracted:
                    triple = item["triple"]
                    level = item["level"]
                    e1, verb, e2 = triple
                    context = sent.get("text", "")
                    rel, rel_conf = self.relation_mapper.classify(e1, verb, e2, context)
                    neighbors = set()
                    for e in sentence_entity_set:
                        if e != e1:
                            neighbors.add(e)
                    if e2 not in neighbors:
                        neighbors.add(e2)
                    resolved_e1 = self.entity_resolver.resolve(e1, neighbors)
                    resolved_e2 = self.entity_resolver.resolve(e2, neighbors)
                    if resolved_e1 == resolved_e2:
                        continue
                    resolution_method = "exact"
                    if e1.lower().strip() != resolved_e1:
                        resolution_method = "surface_form"
                    all_triple_results.append({
                        "triple": (resolved_e1, rel, resolved_e2),
                        "level": level,
                        "relation_confidence": rel_conf,
                        "resolution_method": resolution_method,
                        "doc_index": doc_idx,
                    })
        elapsed = time.time() - t_start
        logger.info("Extraction complete: %d docs in %.1fs, %d raw triples",
                     total, elapsed, len(all_triple_results))
        graph_data = self.graph_builder.build(all_triple_results)
        result = {
            "graph_data": graph_data,
            "stats": {
                "docs_processed": total,
                "raw_triples": len(all_triple_results),
                "kept_triples": graph_data["edge_count"],
                "nodes": graph_data["node_count"],
                "edges": graph_data["edge_count"],
                "relation_types": graph_data["relation_types"],
                "time_seconds": round(elapsed, 2),
                "triples_per_second": round(len(all_triple_results) / elapsed, 1) if elapsed > 0 else 0,
            },
            "entity_resolver": self.entity_resolver,
        }
        return result

    def build_graph_store(self, graph_data: Dict[str, Any]) -> Any:
        concepts = graph_data["concepts"]
        edges = graph_data["edges"]
        id_to_label = graph_data["id_to_label"]
        embeddings = graph_data["embeddings"]
        from graph.graph_component_implementation.dict_graph_store import DictGraphStore
        store = DictGraphStore()
        store.add_dataset(
            concepts=concepts,
            edges=edges,
            id_to_label=id_to_label,
            embeddings=embeddings,
            relation_map=None,
        )
        logger.info("DictGraphStore built: %d nodes, %d edges",
                     store.get_node_count(), store.get_edge_count())
        return store

    def process_and_store(
        self, texts: List[str],
        max_docs: Optional[int] = None,
        show_progress: bool = True,
    ) -> Any:
        result = self.process_corpus(
            texts=texts, show_progress=show_progress, max_docs=max_docs,
        )
        store = self.build_graph_store(result["graph_data"])
        store._stats = result["stats"]
        return store

    def run_with_glmx_feedback(
        self, texts: List[str], glmx_pipeline: Any,
        test_queries: List[str], max_docs: Optional[int] = None,
    ) -> Any:
        store = self.process_and_store(texts=texts, max_docs=max_docs)
        glmx_pipeline.graph_store = store
        for query in test_queries:
            try:
                result = glmx_pipeline.ask(query)
                self.active_learner.observe_failure(
                    query_text=query,
                    template_matched=result.get("template_matched", False),
                    walk_confidence=result.get("walk_confidence", 0.0),
                    path_labels=result.get("walk_path_labels", []),
                    plan_confidence=result.get("confidence", 0.0),
                )
            except Exception as e:
                logger.debug("GLM-X query failed during feedback: %s", e)
        weak = self.active_learner.get_weak_entities(top_k=5)
        if weak:
            logger.info("Weak entities identified: %s", weak)
        return store
