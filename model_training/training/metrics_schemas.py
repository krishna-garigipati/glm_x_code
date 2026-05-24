from typing import Dict, List, Optional, Any


DATALOADER_METRICS_SCHEMA = {
    "source_file": "",
    "format": "",
    "records_loaded": 0,
    "records_after_filter": 0,
    "sentences_extracted": 0,
    "triples_extracted": 0,
    "load_time_seconds": 0.0,
    "parse_error_count": 0,
    "language_distribution": {},
    "filter_reason_counts": {},
}


KGBUILDER_METRICS_SCHEMA = {
    "sentences_processed": 0,
    "spans_extracted": 0,
    "triples_raw": 0,
    "triples_kept": 0,
    "unique_entities_before_resolution": 0,
    "unique_entities_after_resolution": 0,
    "entity_resolution_merge_rate": 0.0,
    "entity_resolution_embedding_merges": 0,
    "embedding_dedup_merges": 0,
    "avg_triple_coherence": 0.0,
    "relation_type_count": 0,
    "triple_coherence_distribution": {},
    "extraction_speed_sentences_per_second": 0.0,
    "build_time_seconds": 0.0,
}


GRAPHSTORE_METRICS_SCHEMA = {
    "node_count": 0,
    "edge_count": 0,
    "relation_types": [],
    "avg_edges_per_node": 0.0,
    "density": 0.0,
    "save_time_seconds": 0.0,
    "load_time_seconds": 0.0,
    "db_file_size_bytes": 0,
    "embedding_dim": 384,
    "has_all_embeddings": False,
    "nodes_without_embeddings": 0,
    "dedup_merge_count": 0,
    "metadata_entries": {},
}


INTENT_FFN_METRICS_SCHEMA = {
    "dataset_name": "",
    "is_continual_training": False,
    "total_samples": 0,
    "train_samples": 0,
    "val_samples": 0,
    "test_samples": 0,
    "num_classes": 0,
    "class_distribution": [],
    "class_names": {},
    "epochs_trained": 0,
    "best_epoch": 0,
    "best_val_loss": 0.0,
    "best_val_top1_accuracy": 0.0,
    "final_train_loss": 0.0,
    "final_train_top1_accuracy": 0.0,
    "training_time_seconds": 0.0,
    "optimizer": "adamw",
    "loss_function": "focal",
    "focal_gamma": 2.0,
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 15,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "test_top1_accuracy": 0.0,
    "test_top3_accuracy": 0.0,
    "test_top5_accuracy": 0.0,
    "test_loss": 0.0,
    "confusion_matrix": [],
    "per_class_metrics": {},
}


T5_DECODER_METRICS_SCHEMA = {
    "t5_model_name": "t5-small",
    "is_continual_training": False,
    "train_samples": 0,
    "val_samples": 0,
    "test_samples": 0,
    "epochs_trained": 0,
    "best_epoch": 0,
    "best_val_loss": 0.0,
    "training_time_seconds": 0.0,
    "test_bleu_score": 0.0,
    "test_exact_match_rate": 0.0,
    "test_rouge_l_f1": 0.0,
    "val_perplexity": 0.0,
    "test_perplexity": 0.0,
    "learning_rate": 0.0003,
    "batch_size": 8,
}


WALKER_METRICS_SCHEMA = {
    "total_walks_attempted": 0,
    "walks_completed": 0,
    "walks_successful": 0,
    "walk_completion_rate": 0.0,
    "walk_success_rate": 0.0,
    "avg_path_length": 0.0,
    "avg_path_confidence": 0.0,
    "avg_final_activation": 0.0,
    "dead_end_rate": 0.0,
    "intent_bias_entropy": 0.0,
    "total_reward_accumulated": 0.0,
    "avg_reward_per_walk": 0.0,
    "bias_update_count": 0,
}


RESONANCE_METRICS_SCHEMA = {
    "total_resonance_calls": 0,
    "avg_activation_energy": 0.0,
    "activation_energy_std": 0.0,
    "avg_convergence_iterations": 0.0,
    "avg_propagation_steps": 0.0,
    "avg_resonance_time_ms": 0.0,
    "nodes_activated_above_threshold_avg": 0.0,
    "tier_used_distribution": {},
}


DECODER_METRICS_SCHEMA = {
    "total_decode_calls": 0,
    "template_matched_count": 0,
    "template_match_rate": 0.0,
    "t5_decoder_success_count": 0,
    "t5_decoder_fallback_rate": 0.0,
    "concatenation_fallback_count": 0,
    "total_fallback_rate": 0.0,
    "output_validity_rate": 0.0,
    "avg_output_length_tokens": 0.0,
    "template_coverage": {},
    "validation_failures": {},
}


QA_EVALUATION_METRICS_SCHEMA = {
    "evaluation_name": "",
    "qa_correct": 0,
    "qa_total": 0,
    "qa_accuracy": 0.0,
    "avg_latency_seconds": 0.0,
    "answer_confidence_mean": 0.0,
    "walk_confidence_mean": 0.0,
    "per_question_results": [],
    "failure_analysis": {},
    "accuracy_by_type": {},
}


CONTINUAL_LEARNING_METRICS_SCHEMA = {
    "previous_dataset_name": "",
    "previous_checkpoint_path": "",
    "forgetting_evaluation": {},
    "transfer_evaluation": {},
    "replay_buffer": {},
    "learning_rate_adjustment": {},
}


COMPONENT_SCHEMAS = {
    "data_loader": DATALOADER_METRICS_SCHEMA,
    "kg_builder": KGBUILDER_METRICS_SCHEMA,
    "graph_store": GRAPHSTORE_METRICS_SCHEMA,
    "intent_ffn": INTENT_FFN_METRICS_SCHEMA,
    "t5_decoder": T5_DECODER_METRICS_SCHEMA,
    "walker": WALKER_METRICS_SCHEMA,
    "resonance": RESONANCE_METRICS_SCHEMA,
    "decoder": DECODER_METRICS_SCHEMA,
    "qa_eval": QA_EVALUATION_METRICS_SCHEMA,
    "continual": CONTINUAL_LEARNING_METRICS_SCHEMA,
}
