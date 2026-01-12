#!/usr/bin/env python3
"""
Run test suites for smollm-diffusion-agent.

Usage:
    python run_tests.py           # Run fast tests only
    python run_tests.py --all     # Run all tests including slow ones
    python run_tests.py --slow    # Run only slow tests
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run smollm-diffusion-agent tests")
    parser.add_argument("--all", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--slow", action="store_true", help="Run only slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    cmd = ["python", "-m", "pytest"]
    
    if args.slow:
        cmd.extend(["-m", "slow"])
    elif not args.all:
        cmd.extend(["-m", "not slow"])
    
    if args.verbose:
        cmd.append("-vv")
    
    cmd.append("tests/")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
