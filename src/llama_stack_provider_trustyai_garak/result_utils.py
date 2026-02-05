"""Shared utilities for parsing Garak scan results.

Pure functions that work with string content - used by both inline and remote implementations.
"""

import json
import re
import logging
from typing import List, Dict, Any, Tuple
from jinja2 import Environment, PackageLoader, select_autoescape
import importlib.resources

logger = logging.getLogger(__name__)


def parse_generations_from_report_content(
    report_content: str,
    eval_threshold: float
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Parse enhanced generations and score rows from report.jsonl content.
    
    Args:
        report_content: String content of report.jsonl file
        eval_threshold: Threshold for determining vulnerability (0-1 scale)
        
    Returns:
        Tuple of (generations, score_rows_by_probe)
        - generations: List of dicts with attempt details
        - score_rows_by_probe: Dict mapping probe_name → list of detector result dicts
    """
    generations = []
    score_rows_by_probe = {}
    
    for line in report_content.split("\n"):
        if not line.strip():
            continue
        
        try:
            entry = json.loads(line)
            
            # Only process completed attempts
            if entry.get("entry_type") == "attempt" and entry.get("status") == 2:
                probe_name = entry.get("probe_classname", "unknown")
                detector_results = entry.get("detector_results", {})
                outputs = entry.get("outputs", [])
                
                # Check if vulnerable based on any detector
                is_vulnerable = False
                for detector, scores in detector_results.items():
                    # Note: scores can be a list (multiple outputs per prompt)
                    scores_list = scores if isinstance(scores, list) else [scores]
                    if any(score >= eval_threshold for score in scores_list):
                        is_vulnerable = True
                        break
                
                # Build enhanced generation
                generation = {
                    "probe": probe_name,
                    "probe_category": probe_name.split('.')[0],
                    "goal": entry.get("goal", ""),
                    "vulnerable": is_vulnerable,
                    "prompt": entry.get("prompt", ""),
                    "responses": outputs,
                    "detector_results": detector_results,
                }
                generations.append(generation)
                
                # Collect score row for this attempt
                if probe_name not in score_rows_by_probe:
                    score_rows_by_probe[probe_name] = []
                
                score_rows_by_probe[probe_name].append(detector_results)
        
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON line in report: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error parsing report line: {e}")
            continue
    
    return generations, score_rows_by_probe


def parse_digest_from_report_content(report_content: str) -> Dict[str, Any]:
    """Parse digest entry from report.jsonl content.
    
    Args:
        report_content: String content of report.jsonl file
        
    Returns:
        Dict with digest data including group and probe summaries, or empty dict if not found
    """
    for line in report_content.split("\n"):
        if not line.strip():
            continue
        
        try:
            entry = json.loads(line)
            
            if entry.get("entry_type") == "digest":
                return entry
        
        except json.JSONDecodeError:
            continue
        except Exception:
            continue
    
    return {}


def parse_aggregated_from_avid_content(avid_content: str) -> Dict[str, Dict[str, Any]]:
    """Parse probe-level aggregated info from AVID report content.
    
    Args:
        avid_content: String content of .avid.jsonl file (can be empty)
        
    Returns:
        Dict mapping probe_name → aggregated_results dict
    """
    if not avid_content:
        return {}
    
    aggregated_by_probe = {}
    
    for line in avid_content.split("\n"):
        if not line.strip():
            continue
        
        try:
            entry = json.loads(line)
            
            # Extract probe name from description
            desc = entry.get("problemtype", {}).get("description", {}).get("value", "")
            probe_match = re.search(r'probe `([^`]+)`', desc)
            probe_name = probe_match.group(1) if probe_match else "unknown"
            
            # Get metrics DataFrame
            metrics_list = entry.get("metrics", [])
            if not metrics_list:
                continue
            
            results = metrics_list[0].get("results", {})
            
            # Parse DataFrame columns to get summary statistics
            
            
            detector_keys = list(results.get("detector", {}).keys())
            if not detector_keys:
                continue
            
            # Use first detector's total (all detectors have same total_attempts)
            first_idx = detector_keys[0]
            total_attempts = results["total_evaluated"][first_idx]
            
            # For benign count, use minimum passed across all detectors
            # (conservative: an attempt is only benign if ALL detectors passed it)
            benign_responses = min(
                results["passed"][idx] for idx in detector_keys
            )
            
            vulnerable_responses = total_attempts - benign_responses
            attack_success_rate = round((vulnerable_responses / total_attempts * 100), 2) if total_attempts > 0 else 0
            
            # Get AVID taxonomy
            impact = entry.get("impact", {}).get("avid", {})
            
            # Get model info
            artifacts = entry.get("affects", {}).get("artifacts", [])
            model_name = artifacts[0].get("name", "unknown") if artifacts else "unknown"
            deployer = entry.get("affects", {}).get("deployer", [])
            model_type = deployer[0] if deployer else "unknown"
            
            # Build aggregated results with clean hierarchy
            aggregated_by_probe[probe_name] = {
                # Core statistics (top level)
                "total_attempts": total_attempts,
                "benign_responses": benign_responses,
                "vulnerable_responses": vulnerable_responses,
                "attack_success_rate": attack_success_rate,
                
                # Metadata (grouped)
                "metadata": {
                    "avid_taxonomy": {
                        "risk_domain": impact.get("risk_domain", []),
                        "sep_view": impact.get("sep_view", []),
                        "lifecycle_view": impact.get("lifecycle_view", [])
                    },
                    "model": {
                        "type": model_type,
                        "name": model_name
                    }
                }
            }
        
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON line in AVID report: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error parsing AVID line: {e}")
            continue
    
    return aggregated_by_probe


def detector_scores_from_digest(
    digest_probe_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Get detector scores from digest.
    
    Args:
        digest_probe_data: Probe-level data from digest
        
    Returns:
        Dict with detector information including scores, defcon levels, etc.
    """
    detectors = {}
    
    # Iterate through all detectors found in the digest probe data
    for detector_name, detector_info in digest_probe_data.items():
        # Skip the _summary key
        if detector_name == "_summary":
            continue
        
        detectors[detector_name] = {
            "detector_descr": detector_info.get("detector_descr", ""),
            "absolute_score": detector_info.get("absolute_score"),
            "absolute_defcon": detector_info.get("absolute_defcon"),
            "absolute_comment": detector_info.get("absolute_comment", ""),
            "total_evaluated": detector_info.get("total_evaluated"),
            "passed": detector_info.get("passed"),
        }
    
    return detectors


def calculate_basic_aggregates(
    score_rows: List[Dict[str, Any]],
    eval_threshold: float
) -> Dict[str, Any]:
    """Calculate basic aggregates when AVID report not available.
    
    Args:
        score_rows: List of detector results per attempt
        eval_threshold: Threshold for determining vulnerability (0-1 scale)
        
    Returns:
        Basic aggregated statistics with same structure as AVID-enhanced results
    """
    total_attempts = len(score_rows)
    vulnerable_count = 0
    
    for row in score_rows:
        # Check if any detector flagged this attempt
        for detector, scores in row.items():
            scores_list = scores if isinstance(scores, list) else [scores]
            if any(score >= eval_threshold for score in scores_list):
                vulnerable_count += 1
                break  # Count attempt once if any detector flagged it
    
    
    attack_success_rate = round((vulnerable_count / total_attempts * 100), 2) if total_attempts > 0 else 0
    
    return {
        # Core statistics
        "total_attempts": total_attempts,
        "benign_responses": total_attempts - vulnerable_count,
        "vulnerable_responses": vulnerable_count,
        "attack_success_rate": attack_success_rate,
        
        # Metadata (minimal when AVID not available)
        "metadata": {}
    }


def calculate_overall_metrics(scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics across all probes.
    
    Args:
        scores: Dict mapping probe_name → score data with aggregated_results
        
    Returns:
        Dict with overall metrics across all probes
    """
    total_attempts = 0
    total_vulnerable = 0
    total_benign = 0
    probe_count = 0
    
    for probe_name, score_data in scores.items():
        aggregated = score_data.get("aggregated_results", {})
        total_attempts += aggregated.get("total_attempts", 0)
        total_vulnerable += aggregated.get("vulnerable_responses", 0)
        total_benign += aggregated.get("benign_responses", 0)
        probe_count += 1
    
    overall_attack_success_rate = round((total_vulnerable / total_attempts * 100), 2) if total_attempts > 0 else 0
    
    return {
        "total_attempts": total_attempts,
        "benign_responses": total_benign,
        "vulnerable_responses": total_vulnerable,
        "attack_success_rate": overall_attack_success_rate,
        "probe_count": probe_count,
    }


def combine_parsed_results(
    generations: List[Dict[str, Any]],
    score_rows_by_probe: Dict[str, List[Dict[str, Any]]],
    aggregated_by_probe: Dict[str, Dict[str, Any]],
    eval_threshold: float,
    digest: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Combine parsed data into EvaluateResponse-compatible structure.
    
    Args:
        generations: List of attempt details
        score_rows_by_probe: Dict mapping probe_name → score rows
        aggregated_by_probe: Dict mapping probe_name → aggregated stats (from AVID)
        eval_threshold: Threshold for vulnerability
        digest: Optional digest data from report.jsonl with detailed probe/detector info
        
    Returns:
        Dict with 'generations' and 'scores' keys (ready for EvaluateResponse)
    """
    scores = {}
    
    # Extract digest eval data if available
    digest_eval = digest.get("eval", {}) if digest else {}
    
    for probe_name, score_rows in score_rows_by_probe.items():
        aggregated = aggregated_by_probe.get(probe_name, {})
        
        # If no AVID data, calculate basic stats from score_rows
        if not aggregated:
            aggregated = calculate_basic_aggregates(score_rows, eval_threshold)
        
        # Get digest data for this probe (navigate through group structure)
        digest_probe_data = None
        if digest_eval:
            # Find the group this probe belongs to (usually first part of probe name)
            probe_group = probe_name.split('.')[0]
            group_data = digest_eval.get(probe_group, {})
            digest_probe_data = group_data.get(probe_name, {})
        
        # Enrich detector scores with digest information if available
        if digest_probe_data:
            detector_scores = detector_scores_from_digest(digest_probe_data)
            aggregated["detector_scores"] = detector_scores
        
        scores[probe_name] = {
            "score_rows": score_rows,
            "aggregated_results": aggregated
        }
    
    # Calculate overall metrics across all probes
    overall_metrics = calculate_overall_metrics(scores)

    # calculate Tier-based Security Aggregate (TBSA) (available from garak>=0.14.0)
    try:
        from garak.analyze import tbsa as tbsa_mod
        tbsa_score, pd_ver_hash, pd_count = tbsa_mod.digest_to_tbsa(digest)
        overall_metrics["tbsa"] = tbsa_score
        overall_metrics["version_probe_hash"] = pd_ver_hash
        overall_metrics["probe_detector_pairs_contributing"] = pd_count
    except Exception as e:
        logger.warning(f"Error calculating tbsa: {e}")
    
    # Add overall metrics as a special entry (using "_overall" to distinguish from probe names)
    scores["_overall"] = {
        "score_rows": [],  # No individual score rows for overall
        "aggregated_results": overall_metrics
    }
    
    return {
        "generations": generations,
        "scores": scores
    }


def parse_jsonl(report_content: str) -> List[Dict[str, Any]]:
    result = []
    for line in report_content.split("\n"):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON line in report: {e}")
            continue
    return result


def vega_data(raw_report: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Grab all the rows that are attempts and have at least one detector result
    filtered_rows = [line
                     for line in raw_report
                     if line.get("entry_type", "") == "attempt"
                     and len(line.get("detector_results", {})) >= 1
                     and line.get("status", 0) == 1]
    
    def parse_detector(detector_map: Dict[str, List[float]]) -> str:
        # If at least one detector returned 1.0 we were successful
        all_scores = [score >= 0.5
                      for scores in detector_map.values()
                      for score in scores[0]]
        return "complied" if any(all_scores) else "refused"
    
    return [{"uuid": row.get("uuid"),
             "probe_classname": row.get("probe_classname"),
             "intent": row.get("notes", {}).get("intent", "unknown_intent"),
             "stub": row.get("notes", {}).get("stub", "unknown_intent"),
             "outcome": parse_detector(row.get("detector_results", {})),
             "detector_results": row.get("detector_results", {}),
             "conversations": row.get("conversations", []),
             "generations": len(row.get("conversations", []))}
            for row in filtered_rows]


def derive_template_vars(raw_report: List[Dict[str, Any]]) -> Dict[str, Any]:
    report_names = [line.get("meta", {}).get("reportfile", "unknown")
                    for line in raw_report
                    if line.get("entry_type", "") == "digest"]
    
    # Load vega_chart_attacks_by_scenario.json from resources folder
    with importlib.resources.files('llama_stack_provider_trustyai_garak.resources').joinpath(
        'vega_chart_attacks_by_scenario.json').open('r') as f:
        vega_chart_attacks_by_scenario = json.load(f)
        # TODO: in this chart we used to pass the list of strategies in input in encoding -> x -> scale -> domain
        #       this way we were able to show attacks that weren't even run because everything complied earlier
        #       we should grab this info from somewhere, possibly the run configuration?...
    
    with importlib.resources.files('llama_stack_provider_trustyai_garak.resources').joinpath(
        'vega_chart_strategy_vs_scenario.json').open('r') as f:
        vega_chart_strategy_vs_scenario = json.load(f)
        # TODO: same as above, pass the list of strategies in input
    
    attacks_by_scenario_data = vega_data(raw_report)
    
    return dict(
        raw_report=raw_report,
        report_name=report_names[0] if report_names else "unknown",
        vega_chart_attacks_by_scenario=vega_chart_attacks_by_scenario,
        vega_chart_strategy_vs_scenario=vega_chart_strategy_vs_scenario,
        attacks_by_scenario_data=attacks_by_scenario_data
    )


def generate_art_report(report_content: str) -> str:
    env = Environment(loader=PackageLoader('llama_stack_provider_trustyai_garak', 'resources'))
    template = env.get_template('art_report.jinja2')
    raw_report = parse_jsonl(report_content)
    template_vars = derive_template_vars(raw_report)
    return template.render(template_vars)
