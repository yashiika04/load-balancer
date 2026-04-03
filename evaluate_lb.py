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
- Saves detailed results to CSV file (results_<algorithm>_<mode>.csv)
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
    
    def __init__(self, lb_url: str, algorithm_name: str, num_trials: int, mode: str = ''):
        """
        Initialize the evaluator.
        
        Args:
            lb_url: Base URL of the load balancer (e.g., http://localhost:5000)
            algorithm_name: Name of the algorithm being tested (for labeling only)
            num_trials: Number of trials to run
            mode: Optional mode or model identifier for file naming
        """
        self.lb_url = lb_url.rstrip('/')
        self.algorithm_name = algorithm_name
        self.num_trials = num_trials
        self.mode = mode.strip() or 'default'
        self.endpoint = f"{self.lb_url}/heavy-task"

    def _fetch_server_metrics(self) -> Dict:
        try:
            response = requests.get(f"{self.lb_url}/server-metrics", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}

    def _compute_additional_metrics(self, details: List[Dict], server_metrics: Dict) -> Tuple[float, float]:
        # Load variance from actual server selection counts during the trial.
        server_counts = {}
        for entry in details:
            server = entry.get('server')
            if server:
                server_counts[server] = server_counts.get(server, 0) + 1

        load_variance = 0.0
        if len(server_counts) > 1:
            load_variance = statistics.pvariance(list(server_counts.values()))

        avg_latency = 0.0
        weighted_latencies = []
        total_weight = 0.0
        for server, entry in server_metrics.items():
            metrics = entry.get('metrics', {})
            total_requests = float(metrics.get('total_requests', 0) or 0)
            avg_latency_s = float(metrics.get('avg_successful_response_time', 0) or 0)
            if total_requests > 0:
                weighted_latencies.append((avg_latency_s, total_requests))
                total_weight += total_requests

        if total_weight > 0 and weighted_latencies:
            avg_latency = sum(lat * weight for lat, weight in weighted_latencies) / total_weight
        elif server_metrics:
            latencies = [float(e.get('metrics', {}).get('avg_successful_response_time', 0) or 0)
                         for e in server_metrics.values()]
            if latencies:
                avg_latency = statistics.mean(latencies)

        return avg_latency, load_variance

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
            details = data.get('details', []) or []
            
            # Calculate success and failure rates
            success_rate = (successes / total_requests * 100) if total_requests > 0 else 0.0
            failure_rate = (failures / total_requests * 100) if total_requests > 0 else 0.0
            throughput = successes / batch_time if batch_time > 0 else 0.0

            # Fetch server-level metrics for latency and load balance analytics
            server_metrics = self._fetch_server_metrics()
            avg_latency, load_variance = self._compute_additional_metrics(details, server_metrics)
            
            print(
                f"    ✓ Completed in {batch_time:.2f}s | Success Rate: {success_rate:.1f}% "
                f"({successes}/{total_requests}) | Failure Rate: {failure_rate:.1f}% | "
                f"Throughput: {throughput:.2f} req/s | Avg Latency: {avg_latency:.3f}s | "
                f"Load Variance: {load_variance:.2f}"
            )
            
            return {
                'trial': trial_num,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'throughput': throughput,
                'avg_latency': avg_latency,
                'load_variance': load_variance,
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
                'failure_rate': 0.0,
                'throughput': 0.0,
                'avg_latency': 0.0,
                'load_variance': 0.0,
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
                'failure_rate': 0.0,
                'throughput': 0.0,
                'avg_latency': 0.0,
                'load_variance': 0.0,
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
        if self.mode and self.mode != 'default':
            print(f"  Mode/Model: {self.mode}")
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
                'failure_rate_mean': 0.0,
                'failure_rate_std': 0.0,
                'throughput_mean': 0.0,
                'throughput_std': 0.0,
                'avg_latency_mean': 0.0,
                'avg_latency_std': 0.0,
                'load_variance_mean': 0.0,
                'load_variance_std': 0.0,
                'batch_time_mean': 0.0,
                'batch_time_std': 0.0
            }
        
        # Extract metrics
        success_rates = [r['success_rate'] for r in valid_results]
        failure_rates = [r['failure_rate'] for r in valid_results]
        throughputs = [r['throughput'] for r in valid_results]
        avg_latencies = [r['avg_latency'] for r in valid_results]
        load_variances = [r['load_variance'] for r in valid_results]
        batch_times = [r['batch_time'] for r in valid_results]
        
        def stats_for(values):
            return {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values)
            }
        
        success_stats = stats_for(success_rates)
        failure_stats = stats_for(failure_rates)
        throughput_stats = stats_for(throughputs)
        latency_stats = stats_for(avg_latencies)
        variance_stats = stats_for(load_variances)
        batch_stats = stats_for(batch_times)
        
        return {
            'trials_completed': len(results),
            'trials_failed': failed_trials,
            'trials_valid': len(valid_results),
            'success_rate_mean': success_stats['mean'],
            'success_rate_std': success_stats['std'],
            'success_rate_min': success_stats['min'],
            'success_rate_max': success_stats['max'],
            'failure_rate_mean': failure_stats['mean'],
            'failure_rate_std': failure_stats['std'],
            'failure_rate_min': failure_stats['min'],
            'failure_rate_max': failure_stats['max'],
            'throughput_mean': throughput_stats['mean'],
            'throughput_std': throughput_stats['std'],
            'throughput_min': throughput_stats['min'],
            'throughput_max': throughput_stats['max'],
            'avg_latency_mean': latency_stats['mean'],
            'avg_latency_std': latency_stats['std'],
            'avg_latency_min': latency_stats['min'],
            'avg_latency_max': latency_stats['max'],
            'load_variance_mean': variance_stats['mean'],
            'load_variance_std': variance_stats['std'],
            'load_variance_min': variance_stats['min'],
            'load_variance_max': variance_stats['max'],
            'batch_time_mean': batch_stats['mean'],
            'batch_time_std': batch_stats['std'],
            'batch_time_min': batch_stats['min'],
            'batch_time_max': batch_stats['max']
        }
    
    def print_summary(self, stats: Dict):
        """Print summary statistics to console."""
        print("\n" + "="*70)
        print("  EVALUATION SUMMARY")
        print("="*70)
        print(f"  Algorithm: {self.algorithm_name}")
        if self.mode and self.mode != 'default':
            print(f"  Mode/Model: {self.mode}")
        print(f"  Total Trials: {stats.get('trials_completed', 0)}")
        
        if stats.get('trials_failed', 0) > 0:
            print(f"  Failed Trials: {stats.get('trials_failed', 0)}")
        
        print(f"  Valid Trials: {stats.get('trials_valid', 0)}")
        print("-"*70)
        
        if stats.get('trials_valid', 0) > 0:
            print("\n  SUCCESS/FAILURE RATES:")
            print(f"    Success Rate Mean: {stats.get('success_rate_mean', 0):.2f}%")
            print(f"    Success Rate Std Dev: {stats.get('success_rate_std', 0):.2f}%")
            print(f"    Failure Rate Mean: {stats.get('failure_rate_mean', 0):.2f}%")
            print(f"    Failure Rate Std Dev: {stats.get('failure_rate_std', 0):.2f}%")
            
            print("\n  THROUGHPUT:")
            print(f"    Mean: {stats.get('throughput_mean', 0):.2f} req/s")
            print(f"    Std Dev: {stats.get('throughput_std', 0):.2f} req/s")
            print(f"    Min: {stats.get('throughput_min', 0):.2f} req/s")
            print(f"    Max: {stats.get('throughput_max', 0):.2f} req/s")
            
            print("\n  AVERAGE LATENCY:")
            print(f"    Mean: {stats.get('avg_latency_mean', 0):.3f} s")
            print(f"    Std Dev: {stats.get('avg_latency_std', 0):.3f} s")
            print(f"    Min: {stats.get('avg_latency_min', 0):.3f} s")
            print(f"    Max: {stats.get('avg_latency_max', 0):.3f} s")
            
            print("\n  LOAD VARIANCE:")
            print(f"    Mean: {stats.get('load_variance_mean', 0):.2f}")
            print(f"    Std Dev: {stats.get('load_variance_std', 0):.2f}")
            print(f"    Min: {stats.get('load_variance_min', 0):.2f}")
            print(f"    Max: {stats.get('load_variance_max', 0):.2f}")
            
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
        filename = f"results_{self.algorithm_name}_{self.mode}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Trial', 'Success Rate (%)', 'Failure Rate (%)', 'Throughput (req/s)',
                'Avg Latency (s)', 'Load Variance', 'Batch Time (s)',
                'Successes', 'Failures', 'Total Requests', 'Error'
            ])
            
            # Write trial data
            for r in results:
                writer.writerow([
                    r['trial'],
                    f"{r['success_rate']:.2f}",
                    f"{r['failure_rate']:.2f}",
                    f"{r['throughput']:.2f}",
                    f"{r['avg_latency']:.4f}",
                    f"{r['load_variance']:.4f}",
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
            writer.writerow(['Mode', self.mode])
            writer.writerow(['Total Trials', stats.get('trials_completed', 0)])
            writer.writerow(['Valid Trials', stats.get('trials_valid', 0)])
            writer.writerow(['Failed Trials', stats.get('trials_failed', 0)])
            writer.writerow([])
            writer.writerow(['Success Rate Mean (%)', f"{stats.get('success_rate_mean', 0):.2f}"])
            writer.writerow(['Success Rate Std Dev (%)', f"{stats.get('success_rate_std', 0):.2f}"])
            writer.writerow(['Success Rate Min (%)', f"{stats.get('success_rate_min', 0):.2f}"])
            writer.writerow(['Success Rate Max (%)', f"{stats.get('success_rate_max', 0):.2f}"])
            writer.writerow(['Failure Rate Mean (%)', f"{stats.get('failure_rate_mean', 0):.2f}"])
            writer.writerow(['Failure Rate Std Dev (%)', f"{stats.get('failure_rate_std', 0):.2f}"])
            writer.writerow(['Failure Rate Min (%)', f"{stats.get('failure_rate_min', 0):.2f}"])
            writer.writerow(['Failure Rate Max (%)', f"{stats.get('failure_rate_max', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Throughput Mean (req/s)', f"{stats.get('throughput_mean', 0):.2f}"])
            writer.writerow(['Throughput Std Dev (req/s)', f"{stats.get('throughput_std', 0):.2f}"])
            writer.writerow(['Throughput Min (req/s)', f"{stats.get('throughput_min', 0):.2f}"])
            writer.writerow(['Throughput Max (req/s)', f"{stats.get('throughput_max', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Avg Latency Mean (s)', f"{stats.get('avg_latency_mean', 0):.4f}"])
            writer.writerow(['Avg Latency Std Dev (s)', f"{stats.get('avg_latency_std', 0):.4f}"])
            writer.writerow(['Avg Latency Min (s)', f"{stats.get('avg_latency_min', 0):.4f}"])
            writer.writerow(['Avg Latency Max (s)', f"{stats.get('avg_latency_max', 0):.4f}"])
            writer.writerow([])
            writer.writerow(['Load Variance Mean', f"{stats.get('load_variance_mean', 0):.4f}"])
            writer.writerow(['Load Variance Std Dev', f"{stats.get('load_variance_std', 0):.4f}"])
            writer.writerow(['Load Variance Min', f"{stats.get('load_variance_min', 0):.4f}"])
            writer.writerow(['Load Variance Max', f"{stats.get('load_variance_max', 0):.4f}"])
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

    parser.add_argument(
        '--mode',
        type=str,
        default='',
        help='Optional identifier for the RL model or evaluation mode (e.g. model4, model5)'
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
        num_trials=args.trials,
        mode=args.mode
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