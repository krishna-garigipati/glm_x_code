#!/usr/bin/env python3
"""
GLM-X Configuration Validation Script

Purpose: Automatically verify that all configuration files are synchronized,
         complete, and ready for parallel team development.

Usage:
    python validate_configs.py --config-dir configs/
    python validate_configs.py --config-dir configs/ --verbose
    python validate_configs.py --config-dir configs/ --check-only vocabulary

Output: PASS/FAIL + detailed report
Exit Code: 0 (all pass), 1 (any fail)

Requires: PyYAML, Python 3.8+
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Tuple, Any
from pathlib import Path


class ConfigValidator:
    """Validates GLM-X configuration files for synchronization."""
    
    def __init__(self, config_dir: str, verbose: bool = False):
        self.config_dir = Path(config_dir)
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
        self.configs = {}
        
    def log(self, msg: str, level: str = "INFO"):
        """Log messages with optional verbosity control."""
        if level == "INFO" and self.verbose:
            print(f"[INFO] {msg}")
        elif level == "WARN":
            print(f"[WARN] {msg}")
            self.warnings.append(msg)
        elif level == "ERROR":
            print(f"[ERROR] {msg}")
            self.errors.append(msg)
        elif level == "PASS":
            print(f"[✓] {msg}")
            self.checks_passed += 1
        elif level == "FAIL":
            print(f"[✗] {msg}")
            self.checks_failed += 1
            self.errors.append(msg)
            
    def load_configs(self) -> bool:
        """Load all YAML configuration files."""
        self.log("Loading configuration files...")
        required_configs = [
            "config_core.yaml",
            "config_graph.yaml",
            "config_resonance.yaml",
            "config_g2p.yaml",
            "config_walker.yaml",
            "config_decoder.yaml",
            "config_learning.yaml",
            "dataclass_schema.yaml",
        ]
        
        all_loaded = True
        for config_name in required_configs:
            config_path = self.config_dir / config_name
            try:
                with open(config_path, 'r') as f:
                    self.configs[config_name] = yaml.safe_load(f)
                    self.log(f"Loaded {config_name}")
            except FileNotFoundError:
                self.log(f"Missing config file: {config_path}", level="FAIL")
                all_loaded = False
            except yaml.YAMLError as e:
                self.log(f"YAML parse error in {config_name}: {e}", level="FAIL")
                all_loaded = False
                
        return all_loaded
    
    def check_intent_vocabulary(self) -> bool:
        """Verify all 16 intents are present and have templates."""
        self.log("\n=== Checking Intent Vocabulary ===", level="INFO")
        
        core = self.configs.get("config_core.yaml", {})
        decoder = self.configs.get("config_decoder.yaml", {})
        
        # Check core has all 16 intents
        intents = core.get("intents", {})
        expected_count = 16
        actual_count = len(intents)
        
        if actual_count != expected_count:
            self.log(f"Intent count mismatch: expected {expected_count}, got {actual_count}",
                    level="FAIL")
            return False
        else:
            self.log(f"Found all {expected_count} intents in config_core.yaml",
                    level="PASS")
        
        # Check decoder has templates for all intents
        templates = decoder.get("templates", {}).get("definitions", [])
        template_intents = set()
        for template in templates:
            intent_seq = template.get("intents", [])
            template_intents.update(intent_seq)
        
        missing_intents = set(range(16)) - template_intents
        if missing_intents:
            self.log(f"Missing templates for intents: {sorted(missing_intents)}",
                    level="FAIL")
            return False
        else:
            self.log(f"All 16 intents have decoder templates", level="PASS")
        
        return True
    
    def check_relation_vocabulary(self) -> bool:
        """Verify all 16 relations are present and have resonance biases."""
        self.log("\n=== Checking Relation Vocabulary ===", level="INFO")
        
        core = self.configs.get("config_core.yaml", {})
        resonance = self.configs.get("config_resonance.yaml", {})
        
        # Check core has all 16 relations
        relations = core.get("relations", {})
        expected_count = 16
        actual_count = len(relations)
        
        if actual_count != expected_count:
            self.log(f"Relation count mismatch: expected {expected_count}, got {actual_count}",
                    level="FAIL")
            return False
        else:
            self.log(f"Found all {expected_count} relations in config_core.yaml",
                    level="PASS")
        
        # Check resonance tier1 has biases for all relations
        tier1_biases = resonance.get("tier1", {}).get("relation_bias", {})
        expected_relations = set(relations.values())
        actual_relations = set(tier1_biases.keys())
        
        missing = expected_relations - actual_relations
        if missing:
            self.log(f"Missing relation_bias in tier1 for: {sorted(missing)}",
                    level="FAIL")
            return False
        else:
            self.log(f"All 16 relations have tier1 biases", level="PASS")
        
        return True
    
    def check_parameter_synchronization(self) -> bool:
        """Verify critical parameters match across configs."""
        self.log("\n=== Checking Parameter Synchronization ===", level="INFO")
        
        params_to_check = [
            ("config_core.yaml", ["activation", "min"], 0.01, "A_rest"),
            ("config_core.yaml", ["activation", "max"], 1.0, "A_max"),
            ("config_core.yaml", ["activation", "threshold_resonance"], 0.2, "θ_resonance"),
            ("config_learning.yaml", ["hebbian", "alpha"], 0.05, "α"),
            ("config_learning.yaml", ["hebbian", "beta"], 0.02, "β"),
            ("config_learning.yaml", ["hebbian", "eligibility_gamma"], 0.9, "γ"),
            ("config_resonance.yaml", ["tier1", "propagation_threshold"], 0.008, "θ_propagate"),
            ("config_walker.yaml", ["walk", "temperature"], 0.1, "temperature"),
        ]
        
        all_match = True
        for config_name, path, expected, name in params_to_check:
            config = self.configs.get(config_name, {})
            value = config
            for key in path:
                value = value.get(key, None)
                if value is None:
                    break
            
            if value == expected:
                self.log(f"{name}: {value} ✓", level="INFO")
            else:
                self.log(f"{name}: expected {expected}, got {value}",
                        level="FAIL")
                all_match = False
        
        return all_match
    
    def check_embedding_quantization(self) -> bool:
        """Verify embedding quantization is properly documented."""
        self.log("\n=== Checking Embedding Quantization ===", level="INFO")
        
        core = self.configs.get("config_core.yaml", {})
        graph = self.configs.get("config_graph.yaml", {})
        
        # Check core has embedding_regularization section
        if "embedding_regularization" not in core:
            self.log("Missing embedding_regularization section in config_core.yaml",
                    level="FAIL")
            return False
        else:
            self.log("Found embedding_regularization section in config_core.yaml",
                    level="PASS")
        
        # Check graph has quantization note
        node = graph.get("node", {})
        if "embedding_quantization_note" not in node:
            self.log("Missing embedding_quantization_note in config_graph.yaml",
                    level="WARN")
        else:
            self.log("Found embedding_quantization_note in config_graph.yaml",
                    level="PASS")
        
        return True
    
    def check_missing_parameters(self) -> bool:
        """Verify all critical missing parameters from Phase 2 are now present."""
        self.log("\n=== Checking for Critical Missing Parameters ===", level="INFO")
        
        core = self.configs.get("config_core.yaml", {})
        
        critical_sections = {
            "embedding_regularization": ["smoothing_eta", "collapse_threshold", "collapse_repulsion_kappa"],
            "sense_disambiguation": ["lsh_similarity_threshold"],
            "analogy": ["temp_edge_strength"],
        }
        
        all_present = True
        for section, keys in critical_sections.items():
            if section not in core:
                self.log(f"Missing section: {section}", level="FAIL")
                all_present = False
            else:
                section_data = core[section]
                for key in keys:
                    if key not in section_data:
                        self.log(f"Missing parameter: {section}.{key}", level="FAIL")
                        all_present = False
                    else:
                        self.log(f"Found {section}.{key} = {section_data[key]}",
                                level="INFO")
        
        return all_present
    
    def check_theta_vector(self) -> bool:
        """Verify theta vector indexing is consistent and correct."""
        self.log("\n=== Checking Theta Vector Structure ===", level="INFO")
        
        resonance = self.configs.get("config_resonance.yaml", {})
        
        theta_indices = resonance.get("theta_indices", {})
        
        # Check theta_dim
        es_controller = resonance.get("es_controller", {})
        theta_dim = es_controller.get("theta_dim", None)
        if theta_dim != 48:
            self.log(f"Theta dimension mismatch: expected 48, got {theta_dim}",
                    level="FAIL")
            return False
        else:
            self.log(f"Theta dimension = 48 ✓", level="PASS")
        
        # Check theta_indices structure
        if "relation_bias_start" not in theta_indices or "relation_bias_end" not in theta_indices:
            self.log("Missing theta_indices.relation_bias_start or relation_bias_end",
                    level="FAIL")
            return False
        
        start = theta_indices.get("relation_bias_start", 0)
        end = theta_indices.get("relation_bias_end", 0)
        expected_relations = 16
        actual_relations = end - start
        
        if actual_relations != expected_relations:
            self.log(f"Relation bias indices mismatch: expected {expected_relations} slots ({start}:{end}), got {actual_relations}",
                    level="FAIL")
            return False
        else:
            self.log(f"Relation bias indices [{start}:{end}] = {expected_relations} relations ✓",
                    level="PASS")
        
        return True
    
    def check_dataclass_schema(self) -> bool:
        """Verify dataclass_schema.yaml is complete and valid."""
        self.log("\n=== Checking Dataclass Schema ===", level="INFO")
        
        schema = self.configs.get("dataclass_schema.yaml", {})
        dataclass_schemas = schema.get("dataclass_schemas", {})
        
        required_dataclasses = ["Subgraph", "Plan", "WalkResult", "Answer", "Node", "Edge"]
        all_present = True
        
        for dc_name in required_dataclasses:
            if dc_name in dataclass_schemas:
                self.log(f"Found dataclass: {dc_name}", level="INFO")
            else:
                self.log(f"Missing dataclass: {dc_name}", level="FAIL")
                all_present = False
        
        if all_present:
            self.log(f"All {len(required_dataclasses)} core dataclasses present",
                    level="PASS")
        
        return all_present
    
    def check_api_interfaces(self) -> bool:
        """Verify all component configs have api_interface sections."""
        self.log("\n=== Checking API Interface Documentation ===", level="INFO")
        
        team_configs = [
            ("config_resonance.yaml", "Resonance"),
            ("config_g2p.yaml", "G2P"),
            ("config_walker.yaml", "Walker"),
            ("config_decoder.yaml", "Decoder"),
            ("config_learning.yaml", "Learning"),
            ("config_graph.yaml", "Graph"),
        ]
        
        all_present = True
        for config_name, team_name in team_configs:
            config = self.configs.get(config_name, {})
            if "api_interface" in config:
                self.log(f"{team_name} has api_interface section", level="INFO")
            else:
                self.log(f"{team_name} missing api_interface section", level="FAIL")
                all_present = False
        
        if all_present:
            self.log(f"All {len(team_configs)} component configs have api_interface",
                    level="PASS")
        
        return all_present
    
    def run_all_checks(self) -> bool:
        """Execute all validation checks."""
        print("\n" + "="*70)
        print("GLM-X CONFIGURATION VALIDATION")
        print("="*70)
        
        if not self.load_configs():
            print("\n[FAIL] Could not load all configuration files")
            return False
        
        results = {
            "Intent Vocabulary": self.check_intent_vocabulary(),
            "Relation Vocabulary": self.check_relation_vocabulary(),
            "Parameter Synchronization": self.check_parameter_synchronization(),
            "Embedding Quantization": self.check_embedding_quantization(),
            "Critical Parameters": self.check_missing_parameters(),
            "Theta Vector": self.check_theta_vector(),
            "Dataclass Schema": self.check_dataclass_schema(),
            "API Interfaces": self.check_api_interfaces(),
        }
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        for check_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {check_name}")
        
        total_checks = len(results)
        passed_checks = sum(1 for r in results.values() if r)
        
        print(f"\nTotal: {passed_checks}/{total_checks} checks passed")
        
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n" + "="*70)
            print("✓ ALL VALIDATIONS PASSED - READY FOR PARALLEL DEVELOPMENT")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ VALIDATION FAILED - FIX ISSUES BEFORE PROCEEDING")
            print("="*70)
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate GLM-X configuration synchronization"
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Path to configs directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--check-only",
        help="Run only specific check (vocabulary, parameters, theta, schema, interfaces)"
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.config_dir, verbose=args.verbose)
    success = validator.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
