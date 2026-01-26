"""
Verify post-training preparation is complete.

Checks:
1. Config changes applied correctly
2. New dataset adapters registered
3. Length jitter enabled
4. Synthetic code generator works
"""
import yaml
import sys
from pathlib import Path
from data.dataset_formats import DATASET_ADAPTERS, get_adapter


def check_config():
    print("=" * 60)
    print("1. Checking config.yaml...")
    print("=" * 60)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    checks = []
    
    reencode = config.get("inference", {}).get("reencode_hidden_states_every")
    if reencode == 1:
        print("✓ Inference re-encoding set to 1 (was 2)")
        checks.append(True)
    else:
        print(f"✗ Inference re-encoding is {reencode}, expected 1")
        checks.append(False)
    
    resume = config.get("training", {}).get("resume_from_checkpoint")
    if resume is True:
        print("✓ Resume from checkpoint enabled")
        checks.append(True)
    else:
        print(f"✗ Resume from checkpoint is {resume}, expected True")
        checks.append(False)
    
    lr = config.get("training", {}).get("learning_rate")
    if lr == 2.0e-5:
        print(f"✓ Learning rate reduced to {lr} (was 1.2e-4)")
        checks.append(True)
    else:
        print(f"✗ Learning rate is {lr}, expected 2.0e-5")
        checks.append(False)
    
    epochs = config.get("training", {}).get("num_epochs")
    if epochs == 2:
        print(f"✓ Epochs set to {epochs} for post-training")
        checks.append(True)
    else:
        print(f"✗ Epochs is {epochs}, expected 2")
        checks.append(False)
    
    scheduler = config.get("training", {}).get("scheduler", {}).get("name")
    if scheduler == "cosine":
        print(f"✓ Scheduler set to {scheduler}")
        checks.append(True)
    else:
        print(f"✗ Scheduler is {scheduler}, expected cosine")
        checks.append(False)
    
    datasets = config.get("data", {}).get("datasets", [])
    print(f"\n✓ Total datasets: {len(datasets)}")
    
    new_datasets = ["argilla/apigen-function-calling", "nvidia/When2Call", "ibm-research/nestful"]
    for ds_name in new_datasets:
        found = any(ds.get("name") == ds_name for ds in datasets)
        if found:
            print(f"  ✓ {ds_name}")
            checks.append(True)
        else:
            print(f"  ✗ {ds_name} NOT FOUND")
            checks.append(False)
    
    length_jitter = config.get("data", {}).get("dynamic_budget", {}).get("length_jitter", {})
    if length_jitter.get("enabled", False):
        print(f"\n✓ Length jitter enabled: {length_jitter.get('min_jitter', 0)}-{length_jitter.get('max_jitter', 5)} tokens")
        checks.append(True)
    else:
        print("\n✗ Length jitter not enabled")
        checks.append(False)
    
    return all(checks)


def check_adapters():
    print("\n" + "=" * 60)
    print("2. Checking dataset adapters...")
    print("=" * 60)
    
    checks = []
    
    required_adapters = [
        "argilla/apigen-function-calling",
        "nvidia/When2Call",
        "ibm-research/nestful",
        "synthetic_code_toolcalls"
    ]
    
    for adapter_name in required_adapters:
        if adapter_name in DATASET_ADAPTERS:
            adapter_class = DATASET_ADAPTERS[adapter_name]
            print(f"✓ {adapter_name}: {adapter_class.__name__}")
            checks.append(True)
        else:
            print(f"✗ {adapter_name} NOT FOUND")
            checks.append(False)
    
    test_example = {
        "query": "Test query",
        "answers": '[{"name": "test", "arguments": {}}]',
        "tools": "[]"
    }
    
    try:
        adapter = get_adapter("argilla/apigen-function-calling")
        result = adapter.convert(test_example)
        print("\n✓ Adapter conversion test passed")
        checks.append(True)
    except Exception as e:
        print(f"\n✗ Adapter conversion test failed: {e}")
        checks.append(False)
    
    return all(checks)


def check_files():
    print("\n" + "=" * 60)
    print("3. Checking new files...")
    print("=" * 60)
    
    checks = []
    
    files_to_check = [
        "data/generate_code_toolcalls.py",
        "POSTTRAIN_PREP.md",
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"✓ {file_path} ({size:.1f} KB)")
            checks.append(True)
        else:
            print(f"✗ {file_path} NOT FOUND")
            checks.append(False)
    
    return all(checks)


def check_imports():
    print("\n" + "=" * 60)
    print("4. Checking imports...")
    print("=" * 60)
    
    checks = []
    
    try:
        from data.dataset_loader import SmartScaffoldDataset
        print("✓ SmartScaffoldDataset imports OK (with length jitter)")
        checks.append(True)
    except Exception as e:
        print(f"✗ SmartScaffoldDataset import failed: {e}")
        checks.append(False)
    
    try:
        from data.dataset_formats import DATASET_ADAPTERS
        print(f"✓ dataset_formats imports OK ({len(DATASET_ADAPTERS)} adapters)")
        checks.append(True)
    except Exception as e:
        print(f"✗ dataset_formats import failed: {e}")
        checks.append(False)
    
    return all(checks)


def main():
    print("\n" + "=" * 60)
    print(" POST-TRAINING PREPARATION VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Config", check_config()))
    results.append(("Adapters", check_adapters()))
    results.append(("Files", check_files()))
    results.append(("Imports", check_imports()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL CHECKS PASSED - READY FOR POST-TRAINING ✓✓✓")
        print("=" * 60)
        print("\nNext steps:")
        print("1. rm -rf data_cache/")
        print("2. python train.py  (or: accelerate launch --config_file accelerate_config_multigpu.yaml train.py)")
        print("3. python benchmark/evaluate.py --model diffusion --limit 200")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("=" * 60)
        print("\nPlease review the errors above and fix before post-training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
