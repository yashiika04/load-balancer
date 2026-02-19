#!/usr/bin/env python3
"""
Load Balancer Performance Evaluation Script
============================================
This script evaluates load balancing algorithms by running multiple trials
and computing statistical metrics (mean, standard deviation) for:
- Success Rate (%)
- Batch Response Time (seconds)

Usage:
------
1. Start your load balancer with the desired algorithm:
   - For Round Robin: set LB_ALGO=RoundRobin in .env and run loadBalancer.py
   - For RL Agent: set LB_ALGO=RLAgent in .env and run loadBalancer.py

2. Run this script:
   python evaluate_lb.py --algorithm RoundRobin --trials 10
   
   Or with custom URL:
   python evaluate_lb.py --url http://localhost:8005 --algorithm RLAgent --trials 15

Output:
-------
- Prints summary statistics to console
- Saves detailed results to CSV file (results_<algorithm>.csv)
"""

import argparse
import requests
import time
import statistics
import csv
import os
from typing import List, Dict, Tuple

class LoadBalancerEvaluator:
    """Evaluates load balancing algorithm performance over multiple trials."""
    
    def __init__(self, lb_url: str, algorithm_name: str, num_trials: int):
        """
        Initialize the evaluator.
        
        Args:
            lb_url: Base URL of the load balancer (e.g., http://localhost:5000)
            algorithm_name: Name of the algorithm being tested (for labeling only)
            num_trials: Number of trials to run
        """
        self.lb_url = lb_url.rstrip('/')
        self.algorithm_name = algorithm_name
        self.num_trials = num_trials
        self.endpoint = f"{self.lb_url}/heavy-task"
        
    def run_single_trial(self, trial_num: int) -> Dict:
        """
        Run a single trial against the load balancer.
        
        Args:
            trial_num: Trial number (for logging)
            
        Returns:
            Dictionary with trial results: {
                'trial': int,
                'success_rate': float,
                'batch_time': float,
                'successes': int,
                'failures': int,
                'total_requests': int,
                'error': str or None
            }
        """
        print(f"  Trial {trial_num}/{self.num_trials}: Sending request to {self.endpoint}...")
        
        start_time = time.time()
        
        try:
            response = requests.get(self.endpoint, timeout=600) #-> time out to handle large requests
            batch_time = time.time() - start_time
            
            if response.status_code != 200:
                return {
                    'trial': trial_num,
                    'success_rate': 0.0,
                    'batch_time': batch_time,
                    'successes': 0,
                    'failures': 0,
                    'total_requests': 0,
                    'error': f"HTTP {response.status_code}"
                }
            
            data = response.json()
            
            # Extract metrics from response
            successes = data.get('successes', 0)
            failures = data.get('failures', 0)
            total_requests = data.get('total_requests', 0)
            
            # Calculate success rate
            success_rate = (successes / total_requests * 100) if total_requests > 0 else 0.0
            
            print(f"    ✓ Completed in {batch_time:.2f}s | Success Rate: {success_rate:.1f}% ({successes}/{total_requests})")
            
            return {
                'trial': trial_num,
                'success_rate': success_rate,
                'batch_time': batch_time,
                'successes': successes,
                'failures': failures,
                'total_requests': total_requests,
                'error': None
            }
            
        except requests.exceptions.Timeout:
            batch_time = time.time() - start_time
            print(f"    ✗ Request timed out after {batch_time:.2f}s")
            return {
                'trial': trial_num,
                'success_rate': 0.0,
                'batch_time': batch_time,
                'successes': 0,
                'failures': 0,
                'total_requests': 0,
                'error': 'Timeout'
            }
            
        except requests.exceptions.RequestException as e:
            batch_time = time.time() - start_time
            print(f"    ✗ Request failed: {str(e)}")
            return {
                'trial': trial_num,
                'success_rate': 0.0,
                'batch_time': batch_time,
                'successes': 0,
                'failures': 0,
                'total_requests': 0,
                'error': str(e)
            }
    
    def run_evaluation(self) -> List[Dict]:
        """
        Run all trials and collect results.
        
        Returns:
            List of trial result dictionaries
        """
        print("\n" + "="*70)
        print(f"  Load Balancer Performance Evaluation")
        print("="*70)
        print(f"  Algorithm: {self.algorithm_name}")
        print(f"  Load Balancer URL: {self.lb_url}")
        print(f"  Number of Trials: {self.num_trials}")
        print("="*70 + "\n")
        
        # Check if load balancer is accessible
        try:
            health_response = requests.get(f"{self.lb_url}/health-check", timeout=5)
            if health_response.status_code == 200:
                print("✓ Load balancer is accessible\n")
            else:
                print(f"⚠ Load balancer returned status {health_response.status_code}\n")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not reach load balancer: {e}")
            print("  Make sure the load balancer is running!\n")
        
        # Run all trials
        results = []
        for i in range(1, self.num_trials + 1):
            result = self.run_single_trial(i)
            results.append(result)
            
            # Add a small delay between trials to avoid overwhelming the system
            if i < self.num_trials:
                time.sleep(1)
        
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate summary statistics from trial results.
        
        Args:
            results: List of trial result dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        # Filter out failed trials (with errors)
        valid_results = [r for r in results if r['error'] is None]
        failed_trials = len(results) - len(valid_results)
        
        if not valid_results:
            return {
                'trials_completed': len(results),
                'trials_failed': failed_trials,
                'success_rate_mean': 0.0,
                'success_rate_std': 0.0,
                'batch_time_mean': 0.0,
                'batch_time_std': 0.0
            }
        
        # Extract metrics
        success_rates = [r['success_rate'] for r in valid_results]
        batch_times = [r['batch_time'] for r in valid_results]
        
        # Calculate statistics
        return {
            'trials_completed': len(results),
            'trials_failed': failed_trials,
            'trials_valid': len(valid_results),
            'success_rate_mean': statistics.mean(success_rates),
            'success_rate_std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
            'success_rate_min': min(success_rates),
            'success_rate_max': max(success_rates),
            'batch_time_mean': statistics.mean(batch_times),
            'batch_time_std': statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0,
            'batch_time_min': min(batch_times),
            'batch_time_max': max(batch_times)
        }
    
    def print_summary(self, stats: Dict):
        """Print summary statistics to console."""
        print("\n" + "="*70)
        print("  EVALUATION SUMMARY")
        print("="*70)
        print(f"  Algorithm: {self.algorithm_name}")
        print(f"  Total Trials: {stats.get('trials_completed', 0)}")
        
        if stats.get('trials_failed', 0) > 0:
            print(f"  Failed Trials: {stats.get('trials_failed', 0)}")
        
        print(f"  Valid Trials: {stats.get('trials_valid', 0)}")
        print("-"*70)
        
        if stats.get('trials_valid', 0) > 0:
            print("\n  SUCCESS RATE:")
            print(f"    Mean:    {stats.get('success_rate_mean', 0):.2f}%")
            print(f"    Std Dev: {stats.get('success_rate_std', 0):.2f}%")
            print(f"    Min:     {stats.get('success_rate_min', 0):.2f}%")
            print(f"    Max:     {stats.get('success_rate_max', 0):.2f}%")
            
            print("\n  BATCH RESPONSE TIME:")
            print(f"    Mean:    {stats.get('batch_time_mean', 0):.2f} seconds")
            print(f"    Std Dev: {stats.get('batch_time_std', 0):.2f} seconds")
            print(f"    Min:     {stats.get('batch_time_min', 0):.2f} seconds")
            print(f"    Max:     {stats.get('batch_time_max', 0):.2f} seconds")
        else:
            print("\n  ⚠ No valid trials completed!")
        
        print("="*70 + "\n")
    
    def save_to_csv(self, results: List[Dict], stats: Dict):
        """
        Save detailed results to CSV file.
        
        Args:
            results: List of trial results
            stats: Summary statistics
        """
        filename = f"results_{self.algorithm_name}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Trial', 'Success Rate (%)', 'Batch Time (s)', 
                           'Successes', 'Failures', 'Total Requests', 'Error'])
            
            # Write trial data
            for r in results:
                writer.writerow([
                    r['trial'],
                    f"{r['success_rate']:.2f}",
                    f"{r['batch_time']:.2f}",
                    r['successes'],
                    r['failures'],
                    r['total_requests'],
                    r['error'] or ''
                ])
            
            # Write summary statistics
            writer.writerow([])
            writer.writerow(['SUMMARY STATISTICS'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Algorithm', self.algorithm_name])
            writer.writerow(['Total Trials', stats.get('trials_completed', 0)])
            writer.writerow(['Valid Trials', stats.get('trials_valid', 0)])
            writer.writerow(['Failed Trials', stats.get('trials_failed', 0)])
            writer.writerow([])
            writer.writerow(['Success Rate Mean (%)', f"{stats.get('success_rate_mean', 0):.2f}"])
            writer.writerow(['Success Rate Std Dev (%)', f"{stats.get('success_rate_std', 0):.2f}"])
            writer.writerow(['Success Rate Min (%)', f"{stats.get('success_rate_min', 0):.2f}"])
            writer.writerow(['Success Rate Max (%)', f"{stats.get('success_rate_max', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Batch Time Mean (s)', f"{stats.get('batch_time_mean', 0):.2f}"])
            writer.writerow(['Batch Time Std Dev (s)', f"{stats.get('batch_time_std', 0):.2f}"])
            writer.writerow(['Batch Time Min (s)', f"{stats.get('batch_time_min', 0):.2f}"])
            writer.writerow(['Batch Time Max (s)', f"{stats.get('batch_time_max', 0):.2f}"])
        
        print(f"✓ Results saved to: {filename}\n")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate Load Balancing Performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Round Robin with default settings (10 trials)
  python evaluate_lb.py --algorithm RoundRobin
  
  # Test RL Agent with 15 trials
  python evaluate_lb.py --algorithm RLAgent --trials 15
  
  # Test with custom load balancer URL
  python evaluate_lb.py --url http://localhost:8005 --algorithm RoundRobin --trials 10

Note: Make sure your load balancer is running with the correct LB_ALGO setting before running this script!
        """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8005',
        help='Load balancer base URL (default: http://localhost:8005)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='RoundRobin',
        help='Algorithm name for labeling (e.g., RoundRobin, LeastConnection, RLAgent)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials to run (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.trials < 1:
        print("Error: Number of trials must be at least 1")
        return
    
    # Create evaluator and run
    evaluator = LoadBalancerEvaluator(
        lb_url=args.url,
        algorithm_name=args.algorithm,
        num_trials=args.trials
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Calculate statistics
    stats = evaluator.calculate_statistics(results)
    
    # Print summary
    evaluator.print_summary(stats)
    
    # Save to CSV
    evaluator.save_to_csv(results, stats)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()