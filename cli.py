"""Command-line interface for AI PR Review Assistant.

Usage:
    python -m src --pr 42                      # Review a GitHub PR
    python -m src --diff-file changes.diff     # Review a local diff file
    python -m src --version                    # Show version
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .agent_orchestrator import AgentOrchestrator, ReviewMode
from .context_builder import ReviewContext
from .diff_parser import DiffParser
from .formatters.markdown import MarkdownFormatter
from .formatters.json_formatter import JsonFormatter
from .github_client import GitHubClient
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_env() -> list[str]:
    """Check required environment variables and return missing ones."""
    missing = []

    if not os.getenv("GITHUB_TOKEN"):
        missing.append("GITHUB_TOKEN (GitHub personal access token)")

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY (LLM API key)")

    if not os.getenv("GITHUB_REPOSITORY") and not os.getenv("CI"):
        # GITHUB_REPOSITORY is not strictly required if --repo is passed
        pass

    return missing


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-pr-review",
        description="🤖 AI PR Review Assistant - Automated code review powered by LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src --pr 42                          # Review PR #42 with default settings
  python -m src --pr 42 --mode deep              # Deep review (rules + all AI agents)
  python -m src --pr 42 --mode quick             # Quick review (rules only, no AI)
  python -m src --diff-file changes.diff         # Review a local diff file (no GitHub)
  python -m src --diff-file changes.diff --mode security  # Security review of local diff
  python -m src --pr 42 --output json           # Output results as JSON
  python -m src --pr 42 --output markdown --save review.md  # Save to file
  python -m src --version                        # Show version
  python -m src --check                          # Check environment configuration
        """,
    )

    # Review target
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--pr", "-p",
        type=int,
        metavar="NUMBER",
        help="GitHub PR number to review",
    )
    target_group.add_argument(
        "--diff-file", "-f",
        type=str,
        metavar="FILE",
        help="Path to a local unified diff file to review",
    )
    target_group.add_argument(
        "--check",
        action="store_true",
        help="Check environment configuration and exit",
    )
    target_group.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version information and exit",
    )

    # Review options
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["quick", "standard", "security", "architecture", "qa", "deep"],
        default="standard",
        help="Review mode (default: standard)",
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        metavar="OWNER/REPO",
        help="GitHub repository (default: from GITHUB_REPOSITORY env)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        choices=["markdown", "json", "compact"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        metavar="FILE",
        help="Save output to file instead of stdout",
    )

    # Config options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "deepseek"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model name (overrides LLM_MODEL env)",
    )
    parser.add_argument(
        "--pr-title",
        type=str,
        help="PR title (for --diff-file mode)",
    )
    parser.add_argument(
        "--pr-author",
        type=str,
        help="PR author (for --diff-file mode)",
    )

    # Debug options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel agent execution",
    )

    return parser


def run_review_from_github(
    pr_number: int,
    repo_name: str,
    mode: ReviewMode,
    llm_client: LLMClient,
    no_parallel: bool,
) -> tuple[str, dict]:
    """Run a review from a GitHub PR.

    Args:
        pr_number: PR number.
        repo_name: Repository in owner/repo format.
        mode: Review mode.
        llm_client: Configured LLM client.
        no_parallel: Disable parallel execution.

    Returns:
        Tuple of (formatted_output, raw_result_dict).
    """
    print(f"🔍 Fetching PR #{pr_number} from {repo_name}...")
    gh = GitHubClient(repo_name=repo_name)
    pr_info = gh.get_pr_info(pr_number)
    diff_text = gh.get_pr_diff(pr_number)

    print(f"   Title: {pr_info.title}")
    print(f"   Author: {pr_info.author}")
    print(f"   Files: {pr_info.changed_files} (+{pr_info.additions} -{pr_info.deletions})")

    # Parse diff
    parser = DiffParser()
    file_diffs = parser.parse(diff_text)
    diff_summary = parser.get_summary(file_diffs)

    # Build context
    context = ReviewContext(
        pr_title=pr_info.title,
        pr_body=pr_info.body,
        pr_author=pr_info.author,
        files=[],
        diff_summary=diff_summary,
        file_diffs=file_diffs,
        raw_diff=diff_text,
    )

    # Run review
    print(f"\n🤖 Running {mode.value} review...")
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        mode=mode,
        parallel=not no_parallel,
    )
    result = orchestrator.review(diff_text, context)

    return result.summary, result.to_dict()


def run_review_from_file(
    diff_path: str,
    mode: ReviewMode,
    llm_client: LLMClient | None,
    pr_title: str | None = None,
    pr_author: str | None = None,
    no_parallel: bool = False,
) -> tuple[str, dict]:
    """Run a review from a local diff file.

    Args:
        diff_path: Path to a unified diff file.
        mode: Review mode.
        llm_client: LLM client (can be None for quick mode).
        pr_title: Optional PR title.
        pr_author: Optional PR author.
        no_parallel: Disable parallel execution.

    Returns:
        Tuple of (formatted_output, raw_result_dict).
    """
    diff_file = Path(diff_path)
    if not diff_file.exists():
        raise FileNotFoundError(f"Diff file not found: {diff_path}")

    print(f"📂 Reading diff from: {diff_path}")
    diff_text = diff_file.read_text(encoding="utf-8")

    # Parse diff
    parser = DiffParser()
    file_diffs = parser.parse(diff_text)
    diff_summary = parser.get_summary(file_diffs)

    print(f"   Files: {diff_summary['total_files']}")
    print(f"   Changes: +{diff_summary['total_additions']} -{diff_summary['total_deletions']}")

    # Build context
    context = ReviewContext(
        pr_title=pr_title or f"Local diff review ({diff_path})",
        pr_body="",
        pr_author=pr_author or "local",
        files=[],
        diff_summary=diff_summary,
        file_diffs=file_diffs,
        raw_diff=diff_text,
    )

    # Run review
    print(f"\n🤖 Running {mode.value} review...")
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        mode=mode,
        parallel=not no_parallel,
    )
    result = orchestrator.review(diff_text, context)

    return result.summary, result.to_dict()


def format_output(result_dict: dict, format_type: str) -> str:
    """Format review result based on output type.

    Args:
        result_dict: Raw result dictionary.
        format_type: Output format (markdown, json, compact).

    Returns:
        Formatted string.
    """
    if format_type == "json":
        formatter = JsonFormatter()
        # Reconstruct result — simplified approach
        return formatter.format_compact(result_dict)

    elif format_type == "compact":
        # Minimal summary
        summary = result_dict.get("summary", {})
        sev = summary.get("severity_counts", {})
        return (
            f"Review: {summary['total_issues']} issues "
            f"({sev.get('critical', 0)}C/{sev.get('high', 0)}H/"
            f"{sev.get('medium', 0)}M/{sev.get('low', 0)}L)"
        )

    else:
        # Return the pre-formatted markdown summary
        return result_dict.get("summary", str(result_dict))


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Fix Unicode on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    # Handle --version
    if args.version:
        print(f"AI PR Review Assistant v{__version__}")
        print(f"Python {sys.version}")
        return 0

    # Handle --check
    if args.check:
        print("🔧 Checking environment configuration...\n")
        missing = check_env()
        if missing:
            print("❌ Missing environment variables:")
            for m in missing:
                print(f"   - {m}")
            print("\nSet them with:")
            print("   export VAR_NAME=value    # Linux/macOS")
            print("   $env:VAR_NAME = 'value'  # PowerShell")
            return 1
        else:
            print("✅ All required environment variables are set!")
            print(f"   LLM Provider: {os.getenv('LLM_PROVIDER', 'openai')}")
            print(f"   LLM Model: {os.getenv('LLM_MODEL', '(default)')}")
            return 0

    # Determine mode
    mode = ReviewMode(args.mode)

    # Initialize LLM client (if needed)
    llm_client = None
    if mode != ReviewMode.QUICK:
        try:
            llm_client = LLMClient.from_env(
                provider=args.provider,
                model=args.model,
            )
        except Exception as e:
            print(f"❌ Failed to initialize LLM client: {e}")
            print("   Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set.")
            return 1

    try:
        # Run review
        if args.pr:
            repo_name = args.repo or os.getenv("GITHUB_REPOSITORY")
            if not repo_name:
                print("❌ Repository is required. Use --repo or set GITHUB_REPOSITORY env.")
                return 1
            output_text, result_dict = run_review_from_github(
                pr_number=args.pr,
                repo_name=repo_name,
                mode=mode,
                llm_client=llm_client,
                no_parallel=args.no_parallel,
            )

        elif args.diff_file:
            output_text, result_dict = run_review_from_file(
                diff_path=args.diff_file,
                mode=mode,
                llm_client=llm_client,
                pr_title=args.pr_title,
                pr_author=args.pr_author,
                no_parallel=args.no_parallel,
            )

        else:
            parser.print_help()
            return 0

        # Format and output
        formatted = format_output(result_dict, args.output)

        if args.save:
            Path(args.save).write_text(formatted, encoding="utf-8")
            print(f"\n✅ Review saved to: {args.save}")
        else:
            print(f"\n{formatted}")

        # Show severity counts
        sev_counts = result_dict.get("severity_counts", {})
        total = result_dict.get("total_issues", 0)
        if total > 0:
            print(
                f"\n📊 {total} issues: "
                f"🔴{sev_counts.get('critical', 0)} "
                f"🟠{sev_counts.get('high', 0)} "
                f"🟡{sev_counts.get('medium', 0)} "
                f"🟢{sev_counts.get('low', 0)}"
            )

        return 0

    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during review")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
