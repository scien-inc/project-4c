"""
roi_analysis_cli.py
Command-line interface for ROI analysis workflow
"""
import argparse
import json
import os
import sys
from typing import Dict, Any

from roi_analysis_workflow import ROIAnalysisWorkflow
from domain.roitree import mermaid_to_roi_tree


def save_mermaid_diagram(mermaid_diagram: str, output_path: str) -> None:
    """
    Save Mermaid diagram to a file
    
    Args:
        mermaid_diagram: Mermaid diagram text
        output_path: Path to save the diagram
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_diagram)
    
    print(f"Mermaid diagram saved to {output_path}")


def save_analysis_results(analysis_results: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis results to a JSON file
    
    Args:
        analysis_results: Analysis results dictionary
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis results saved to {output_path}")


def load_mermaid_diagram(input_path: str) -> str:
    """
    Load Mermaid diagram from a file
    
    Args:
        input_path: Path to the Mermaid diagram file
        
    Returns:
        Mermaid diagram text
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_challenge(args):
    """
    Analyze a business challenge and create an ROI tree
    
    Args:
        args: Command-line arguments
    """
    # Initialize workflow
    workflow = ROIAnalysisWorkflow(model_name=args.model)
    
    # Read challenge from file or command line
    challenge_text = ""
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            challenge_text = f.read()
    else:
        challenge_text = args.challenge
    
    print(f"Analyzing challenge: {challenge_text[:100]}...")
    
    # Analyze challenge
    root_node, mermaid_diagram = workflow.analyze_challenge(challenge_text)
    
    # Save Mermaid diagram
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        save_mermaid_diagram(mermaid_diagram, args.output)
        
        # Save analysis results
        if args.analyze:
            analysis = workflow.collect_numerical_data(root_node, mermaid_diagram)
            analysis_output = args.output.rsplit('.', 1)[0] + '_analysis.json'
            save_analysis_results(analysis.dict(), analysis_output)
    else:
        print("\nGenerated ROI Tree (Mermaid):")
        print(mermaid_diagram)
        
        if args.analyze:
            analysis = workflow.collect_numerical_data(root_node, mermaid_diagram)
            print("\nAnalysis Results:")
            print(f"Completion: {analysis.completion_percentage:.1f}%")
            print(f"Has numerical data: {analysis.has_numerical_data}")
            if analysis.missing_nodes:
                print(f"Missing nodes: {', '.join(analysis.missing_nodes)}")


def prioritize_nodes(args):
    """
    Prioritize ROI tree nodes
    
    Args:
        args: Command-line arguments
    """
    # Initialize workflow
    workflow = ROIAnalysisWorkflow(model_name=args.model)
    
    # Load Mermaid diagram
    mermaid_diagram = load_mermaid_diagram(args.input)
    
    # Convert to ROI tree
    roi_tree = mermaid_to_roi_tree(mermaid_diagram)
    if not roi_tree:
        print("Error: Could not parse Mermaid diagram")
        return
    
    # Read priorities from file or command line
    priorities = ""
    if args.priorities_file:
        with open(args.priorities_file, 'r', encoding='utf-8') as f:
            priorities = f.read()
    else:
        priorities = args.priorities
    
    print("Prioritizing nodes...")
    
    # Prioritize nodes
    prioritization_result = workflow.prioritize_nodes(roi_tree, priorities)
    
    # Save results
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        save_analysis_results(prioritization_result.dict(), args.output)
    else:
        print("\nPrioritization Results:")
        print(f"Overall strategy: {prioritization_result.overall_strategy}")
        print("\nPrioritized nodes:")
        for node in prioritization_result.prioritized_nodes:
            print(f"- {node.name}: {node.priority_score}/100 - {node.rationale}")


def generate_proposal(args):
    """
    Generate a proposal based on an ROI tree
    
    Args:
        args: Command-line arguments
    """
    # Initialize workflow
    workflow = ROIAnalysisWorkflow(model_name=args.model)
    
    # Load Mermaid diagram
    mermaid_diagram = load_mermaid_diagram(args.input)
    
    # Convert to ROI tree
    roi_tree = mermaid_to_roi_tree(mermaid_diagram)
    if not roi_tree:
        print("Error: Could not parse Mermaid diagram")
        return
    
    # Read guidance from file or command line
    guidance = ""
    if args.guidance_file:
        with open(args.guidance_file, 'r', encoding='utf-8') as f:
            guidance = f.read()
    else:
        guidance = args.guidance
    
    print("Generating proposal...")
    
    # Generate proposal
    proposal_text, proposal_summary = workflow.generate_proposal(roi_tree, guidance)
    
    # Save results
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(proposal_text)
        
        # Save summary
        summary_output = args.output.rsplit('.', 1)[0] + '_summary.json'
        save_analysis_results(proposal_summary.dict(), summary_output)
        
        print(f"Proposal saved to {args.output}")
        print(f"Summary saved to {summary_output}")
    else:
        print("\nGenerated Proposal:")
        print(proposal_text)
        
        print("\nProposal Summary:")
        print(f"Total investment: {proposal_summary.total_investment:,.0f}円")
        print(f"Total benefit: {proposal_summary.total_benefit:,.0f}円")
        print(f"ROI: {proposal_summary.roi_percentage:.1f}%")
        print(f"Implementation timeframe: {proposal_summary.implementation_timeframe}")
        print("Key recommendations:")
        for i, rec in enumerate(proposal_summary.key_recommendations, 1):
            print(f"{i}. {rec}")


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="ROI Analysis Workflow CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze challenge command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a business challenge")
    analyze_parser.add_argument("challenge", nargs="?", help="Business challenge text")
    analyze_parser.add_argument("-f", "--file", help="Read challenge from file")
    analyze_parser.add_argument("-o", "--output", help="Output file for Mermaid diagram")
    analyze_parser.add_argument("-a", "--analyze", action="store_true", help="Also perform leaf node analysis")
    analyze_parser.add_argument("-m", "--model", default="gpt-4o", help="OpenAI model to use")
    
    # Prioritize nodes command
    prioritize_parser = subparsers.add_parser("prioritize", help="Prioritize ROI tree nodes")
    prioritize_parser.add_argument("input", help="Input Mermaid diagram file")
    prioritize_parser.add_argument("-p", "--priorities", default="", help="User priorities text")
    prioritize_parser.add_argument("-pf", "--priorities-file", help="Read priorities from file")
    prioritize_parser.add_argument("-o", "--output", help="Output file for prioritization results")
    prioritize_parser.add_argument("-m", "--model", default="gpt-4o", help="OpenAI model to use")
    
    # Generate proposal command
    proposal_parser = subparsers.add_parser("proposal", help="Generate a proposal")
    proposal_parser.add_argument("input", help="Input Mermaid diagram file")
    proposal_parser.add_argument("-g", "--guidance", default="", help="Proposal guidance text")
    proposal_parser.add_argument("-gf", "--guidance-file", help="Read guidance from file")
    proposal_parser.add_argument("-o", "--output", help="Output file for proposal")
    proposal_parser.add_argument("-m", "--model", default="gpt-4o", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        if not args.challenge and not args.file:
            analyze_parser.error("Either challenge text or --file is required")
        analyze_challenge(args)
    elif args.command == "prioritize":
        prioritize_nodes(args)
    elif args.command == "proposal":
        generate_proposal(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()