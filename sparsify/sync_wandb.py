import os
import subprocess
import time
import argparse
from pathlib import Path

def sync_wandb_runs(wandb_dir, max_retries=1, verbose=True):
    """
    Syncs each offline wandb run directory individually, skipping any that fail.
    
    Args:
        wandb_dir: Path to the wandb directory containing offline runs
        max_retries: Maximum number of retries per run before giving up
        verbose: Whether to print detailed output
    
    Returns:
        tuple: (list of successfully synced runs, list of failed runs)
    """
    wandb_path = Path(wandb_dir).expanduser().resolve()
    if not wandb_path.exists() or not wandb_path.is_dir():
        raise ValueError(f"Directory not found: {wandb_path}")
    
    # Find all offline run directories
    run_dirs = [d for d in wandb_path.iterdir() if d.is_dir() and d.name.startswith("offline-run-")]
    
    if verbose:
        print(f"Found {len(run_dirs)} offline run directories in {wandb_path}")
    
    successful_runs = []
    failed_runs = []
    
    # Try to sync each run individually
    for run_dir in run_dirs:
        run_id = run_dir.name
        if verbose:
            print(f"\nAttempting to sync run: {run_id}")
        
        for attempt in range(max_retries):
            try:
                if verbose and attempt > 0:
                    print(f"Retry {attempt}/{max_retries} for {run_id}")
                
                # Run the wandb sync command for this specific run
                result = subprocess.run(
                    ["wandb", "sync", str(run_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5-minute timeout per run
                    check=False,  # Don't raise exception on non-zero exit
                )
                
                # Check if sync was successful
                if result.returncode == 0:
                    if verbose:
                        print(f"Successfully synced {run_id}")
                    successful_runs.append(run_id)
                    break
                else:
                    if verbose:
                        print(f"Error syncing {run_id}: {result.stderr.strip()}")
                    
                    # Check if this is a "deleted" error, in which case we skip retries
                    if "entity/project/run has been deleted" in result.stderr:
                        if verbose:
                            print(f"Run {run_id} appears to be deleted. Skipping.")
                        failed_runs.append((run_id, "Run was deleted"))
                        break
                        
                    # Only retry if this isn't the last attempt
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Short pause before retry
                    else:
                        failed_runs.append((run_id, result.stderr.strip()))
            
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"Timeout occurred while syncing {run_id}")
                failed_runs.append((run_id, "Timeout"))
                break
            
            except Exception as e:
                if verbose:
                    print(f"Exception during sync of {run_id}: {str(e)}")
                failed_runs.append((run_id, str(e)))
                break
    
    # Report final results
    if verbose:
        print("\n--- Sync Results ---")
        print(f"Successfully synced: {len(successful_runs)}/{len(run_dirs)} runs")
        print(f"Failed runs: {len(failed_runs)}")
        
        if failed_runs:
            print("\nFailed runs details:")
            for run_id, error in failed_runs:
                print(f"- {run_id}: {error[:100]}..." if len(error) > 100 else f"- {run_id}: {error}")
    
    return successful_runs, failed_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync wandb offline runs individually, skipping failures")
    parser.add_argument("wandb_dir", help="Path to the wandb directory containing offline runs")
    parser.add_argument("--retries", type=int, default=1, help="Maximum number of retries per run")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    sync_wandb_runs(args.wandb_dir, max_retries=args.retries, verbose=not args.quiet)