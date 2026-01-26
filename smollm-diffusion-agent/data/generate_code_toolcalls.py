"""
Generate synthetic code-heavy tool-calling examples.

Creates training data where code snippets, shell commands, and file operations
are passed as tool arguments. This teaches the model to handle:
- Multiline strings (code blocks)
- Special characters and escaping
- Long string values
- Structured text (patches, configs)
"""
import json
import random
from typing import Dict, List, Any
from datasets import load_dataset


CODE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory",
                        "default": "."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file to patch"
                    },
                    "patch_content": {
                        "type": "string",
                        "description": "Patch content in unified diff format"
                    }
                },
                "required": ["file_path", "patch_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for code patterns in a codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search"
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to search",
                        "default": [".py"]
                    }
                },
                "required": ["pattern"]
            }
        }
    }
]


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    tool_call = {
        "name": name,
        "arguments": arguments,
    }
    return f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>"


def generate_python_code_example(code_snippet: str, task_description: str) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": task_description},
            {"from": "gpt", "value": format_tool_call("execute_python", {"code": code_snippet})}
        ],
        "tools": json.dumps(CODE_TOOLS_SCHEMA)
    }


def generate_shell_command_example(command: str, task_description: str) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": task_description},
            {"from": "gpt", "value": format_tool_call("run_shell", {"command": command})}
        ],
        "tools": json.dumps(CODE_TOOLS_SCHEMA)
    }


def generate_file_write_example(file_path: str, content: str, task_description: str) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": task_description},
            {"from": "gpt", "value": format_tool_call("write_file", {"path": file_path, "content": content})}
        ],
        "tools": json.dumps(CODE_TOOLS_SCHEMA)
    }


SIMPLE_PYTHON_TASKS = [
    ("print('Hello, World!')", "Write a Python script that prints 'Hello, World!'"),
    ("import math\nprint(math.pi)", "Write Python code to print the value of pi"),
    ("def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))", 
     "Create a Python function to calculate factorial of 5"),
    ("nums = [1, 2, 3, 4, 5]\nprint(sum(nums))", "Write Python code to sum the list [1, 2, 3, 4, 5]"),
    ("with open('data.txt', 'r') as f:\n    print(f.read())", "Write Python code to read and print contents of data.txt"),
    ("import json\ndata = {'name': 'John', 'age': 30}\nprint(json.dumps(data, indent=2))", 
     "Write Python code to create and print a JSON object with name and age"),
    ("for i in range(10):\n    if i % 2 == 0:\n        print(i)", "Print all even numbers from 0 to 9"),
    ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a)\n        a, b = b, a + b\n\nfibonacci(10)", 
     "Generate first 10 Fibonacci numbers"),
]

SIMPLE_SHELL_COMMANDS = [
    ("ls -la", "List all files in the current directory with details"),
    ("git status", "Check the git repository status"),
    ("pip install requests", "Install the requests Python package"),
    ("mkdir -p data/raw", "Create a nested directory data/raw"),
    ("find . -name '*.py' -type f", "Find all Python files in current directory and subdirectories"),
    ("grep -r 'TODO' .", "Search for TODO comments in all files"),
    ("docker ps -a", "List all Docker containers"),
    ("curl -X GET https://api.github.com/users/octocat", "Fetch user data from GitHub API"),
]

SIMPLE_FILE_OPERATIONS = [
    ("config.yaml", "model:\n  name: 'SmolLM'\n  version: '3B'\ntraining:\n  batch_size: 4\n  learning_rate: 0.0001", 
     "Create a YAML configuration file for model training"),
    ("requirements.txt", "torch>=2.0.0\ntransformers>=4.30.0\ndatasets>=2.14.0", 
     "Create a requirements.txt for a PyTorch project"),
    ("README.md", "# My Project\n\nThis is a sample project.\n\n## Installation\n\n```bash\npip install -r requirements.txt\n```", 
     "Create a README file for a project"),
]


def load_code_snippets_from_dataset(dataset_name: str, limit: int = 1000) -> List[Dict[str, str]]:
    examples = []
    try:
        ds = load_dataset(dataset_name, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        for ex in ds:
            if "output" in ex and ex["output"]:
                code = ex["output"]
                instruction = ex.get("instruction", "Write Python code for this task")
                examples.append({"code": code, "task": instruction})
        
        print(f"Loaded {len(examples)} code examples from {dataset_name}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
    
    return examples


def generate_synthetic_code_dataset(num_simple: int = 500, num_from_hf: int = 500) -> List[Dict[str, Any]]:
    all_examples = []
    
    for _ in range(num_simple // len(SIMPLE_PYTHON_TASKS)):
        for code, task in SIMPLE_PYTHON_TASKS:
            all_examples.append(generate_python_code_example(code, task))
    
    for _ in range(num_simple // len(SIMPLE_SHELL_COMMANDS)):
        for cmd, task in SIMPLE_SHELL_COMMANDS:
            all_examples.append(generate_shell_command_example(cmd, task))
    
    for _ in range(num_simple // len(SIMPLE_FILE_OPERATIONS)):
        for path, content, task in SIMPLE_FILE_OPERATIONS:
            all_examples.append(generate_file_write_example(path, content, task))
    
    hf_code_datasets = [
        "iamtarun/python_code_instructions_18k_alpaca",
        "flytech/python-codes-25k",
    ]
    
    for dataset_name in hf_code_datasets:
        code_examples = load_code_snippets_from_dataset(dataset_name, limit=num_from_hf // len(hf_code_datasets))
        for ex in code_examples:
            all_examples.append(generate_python_code_example(ex["code"], ex["task"]))
    
    random.shuffle(all_examples)
    return all_examples


if __name__ == '__main__':
    print("Generating synthetic code tool-call dataset...")
    dataset = generate_synthetic_code_dataset(num_simple=200, num_from_hf=300)
    print(f"Generated {len(dataset)} examples")
    
    output_path = "code_toolcalls_synthetic.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved to {output_path}")
    print(f"\nSample example:")
    print(json.dumps(dataset[0], indent=2))
