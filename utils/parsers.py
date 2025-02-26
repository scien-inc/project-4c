"""
Utilities for parsing LLM outputs and updating the ROI tree
"""
import re
import json
from typing import Dict, List, Any, Optional, Tuple, cast

from langchain_core.messages import AIMessage
from domain.schemas import ROINodeUpdate, ROIAnalysis, ROICalculation, ProposalAnalysis


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text, handling various formats the LLM might output
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON as a dictionary, or None if no valid JSON found
    """
    # Try to find JSON between triple backticks
    json_pattern = r"```(?:json)?(.*?)```"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON between curly braces
    json_pattern = r"\{.*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def parse_roi_node_update(message: AIMessage) -> List[ROINodeUpdate]:
    """
    Parse an AI message to extract ROI node updates
    
    Args:
        message: AI message potentially containing node updates
        
    Returns:
        List of ROINodeUpdate objects
    """
    content = message.content
    
    # Try to extract as JSON first
    json_data = extract_json_from_text(content)
    if json_data:
        # Handle single node update
        if "name" in json_data:
            try:
                return [ROINodeUpdate(**json_data)]
            except Exception:
                pass
            
        # Handle list of node updates
        if isinstance(json_data, list):
            nodes = []
            for item in json_data:
                try:
                    nodes.append(ROINodeUpdate(**item))
                except Exception:
                    continue
            if nodes:
                return nodes
    
    # Fallback: Try to extract structured information with regex
    nodes = []
    
    # Pattern for node definitions
    node_pattern = r"(?:Node|Component|Element|Factor):\s*(.*?)\n(?:.*?Description|Details):\s*(.*?)(?:\n|$)(?:.*?Importance(?:\s*Factor)?:\s*(\d+(?:\.\d+)?%?))?(?:\n|$)(?:.*?Value:\s*([0-9,.]+)(?:\s*[kKmMbB]?))?(?:\n|$)"
    
    matches = re.finditer(node_pattern, content, re.DOTALL)
    for match in matches:
        name = match.group(1).strip()
        details = match.group(2).strip() if match.group(2) else None
        
        # Parse importance factor
        importance_factor = 1.0
        if match.group(3):
            importance_str = match.group(3).strip()
            if importance_str.endswith('%'):
                importance_factor = float(importance_str.rstrip('%')) / 100
            else:
                importance_factor = float(importance_str)
                
        # Parse value
        value = None
        if match.group(4):
            value_str = match.group(4).strip().replace(',', '')
            value = float(value_str)
            
        nodes.append(ROINodeUpdate(
            name=name,
            details=details,
            importance_factor=importance_factor,
            value=value
        ))
    
    return nodes


def parse_reflection_analysis(message: AIMessage) -> Tuple[bool, Optional[ROIAnalysis]]:
    """
    Parse reflection analysis from AI message
    
    Args:
        message: AI message containing self-reflection
        
    Returns:
        Tuple of (success, analysis)
    """
    content = message.content
    
    # Try to extract as JSON first
    json_data = extract_json_from_text(content)
    if json_data:
        try:
            return True, ROIAnalysis(**json_data)
        except Exception:
            pass
    
    # Fallback regex extraction
    deepdive_needed = "continue" in content.lower() or "more exploration" in content.lower()
    reason_pattern = r"(?:Reason|Reasoning|Rationale):\s*(.*?)(?:\n|$)"
    reason_match = re.search(reason_pattern, content, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else "No explicit reason provided"
    
    # Extract completion percentage
    percentage_pattern = r"(\d+)%\s*(?:complete|finished|done)"
    percentage_match = re.search(percentage_pattern, content)
    completion_percentage = float(percentage_match.group(1)) if percentage_match else 50.0
    
    # Extract suggested focus
    focus_pattern = r"(?:focus on|explore|should focus on)\s*[\"']?(.*?)[\"']?(?:[,\.]|$)"
    focus_match = re.search(focus_pattern, content, re.IGNORECASE)
    suggested_focus = focus_match.group(1).strip() if focus_match else None
    
    analysis = ROIAnalysis(
        deepdive_needed=deepdive_needed,
        reason=reason,
        suggested_focus=suggested_focus,
        deepdive_completion_percentage=completion_percentage
    )
    
    return True, analysis


def parse_roi_calculation(message: AIMessage) -> List[ROICalculation]:
    """
    Parse ROI calculations from AI message
    
    Args:
        message: AI message containing ROI calculations
        
    Returns:
        List of ROICalculation objects
    """
    content = message.content
    
    # Try to extract as JSON first
    json_data = extract_json_from_text(content)
    if json_data:
        calculations = []
        
        # Handle single calculation
        if isinstance(json_data, dict) and "estimated_value" in json_data:
            try:
                return [ROICalculation(**json_data)]
            except Exception:
                pass
            
        # Handle list of calculations
        if isinstance(json_data, list):
            for item in json_data:
                try:
                    calculations.append(ROICalculation(**item))
                except Exception:
                    continue
            if calculations:
                return calculations
    
    # Fallback regex extraction
    calculations = []
    
    # Look for value estimations
    value_pattern = r"(?:Estimated|Potential) value:?\s*(?:[$£€])?\s*([0-9,]+(?:\.\d+)?)\s*(?:[kKmMbB])?"
    value_matches = re.finditer(value_pattern, content)
    
    for match in value_matches:
        value_str = match.group(1).replace(',', '')
        value = float(value_str)
        
        # Look for node name in the vicinity (20 lines above the value)
        context = content[:match.start()].split('\n')[-20:]
        context_text = '\n'.join(context)
        
        node_pattern = r"(?:Node|Component|Element|Factor):\s*(.*?)(?:\n|$)"
        node_match = re.search(node_pattern, context_text)
        node_name = node_match.group(1).strip() if node_match else "Unknown"
        
        # Look for confidence
        confidence_pattern = r"(?:Confidence|Certainty):\s*(\d+)%"
        confidence_match = re.search(confidence_pattern, context_text)
        confidence = float(confidence_match.group(1))/100 if confidence_match else 0.7
        
        # Look for assumptions
        assumptions_pattern = r"(?:Assumptions|Assuming):\s*(.*?)(?:\n\n|$)"
        assumptions_match = re.search(assumptions_pattern, context_text, re.DOTALL)
        
        assumptions = []
        if assumptions_match:
            assumptions_text = assumptions_match.group(1)
            assumptions = [a.strip() for a in assumptions_text.split('\n') if a.strip()]
        
        calculations.append(ROICalculation(
            node_id="auto",  # Will be replaced with actual ID later
            name=node_name,
            estimated_value=value,
            confidence=confidence,
            assumptions=assumptions
        ))
    
    return calculations


def parse_proposal_reflection(message: AIMessage) -> Tuple[bool, Optional[ProposalAnalysis]]:
    """
    Parse proposal reflection from AI message
    
    Args:
        message: AI message containing proposal reflection
        
    Returns:
        Tuple of (success, analysis)
    """
    content = message.content
    
    # Try to extract as JSON first
    json_data = extract_json_from_text(content)
    if json_data:
        try:
            return True, ProposalAnalysis(**json_data)
        except Exception:
            pass
    
    # Fallback regex extraction
    proposal_complete = "complete" in content.lower() or "finished" in content.lower()
    
    reason_pattern = r"(?:Reason|Reasoning|Rationale):\s*(.*?)(?:\n|$)"
    reason_match = re.search(reason_pattern, content, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else "No explicit reason provided"
    
    # Extract confidence
    confidence_pattern = r"(?:Confidence|Certainty):\s*(\d+)%"
    confidence_match = re.search(confidence_pattern, content)
    roi_confidence = float(confidence_match.group(1))/100 if confidence_match else 0.7
    
    # Extract missing information
    missing_info = []
    if "missing" in content.lower() or "need" in content.lower():
        missing_pattern = r"(?:Missing information|Still need):\s*(.*?)(?:\n\n|$)"
        missing_match = re.search(missing_pattern, content, re.DOTALL)
        
        if missing_match:
            missing_text = missing_match.group(1)
            items = re.findall(r"[-*]\s*(.*?)(?:\n|$)", missing_text)
            missing_info = [item.strip() for item in items if item.strip()]
    
    analysis = ProposalAnalysis(
        proposal_complete=proposal_complete,
        reason=reason,
        missing_information=missing_info,
        roi_confidence=roi_confidence
    )
    
    return True, analysis