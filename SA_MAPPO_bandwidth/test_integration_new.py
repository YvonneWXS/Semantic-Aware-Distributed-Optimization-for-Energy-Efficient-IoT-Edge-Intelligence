#!/usr/bin/env python
"""
Integration test for SA-MAPPO with DW-DNA modifications
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.env_discrete import DiscreteActionEnv
from envs.env_core import EnvCore

def test_action_space():
    """Test updated action space"""
    env = DiscreteActionEnv()

    print("=== Testing Action Space ===")
    print(f"Action space: {env.action_space}")
    print(f"High bounds: {env.action_space.high}")
    print(f"Low bounds: {env.action_space.low}")

    # Verify dimensions
    assert env.action_space.shape == 4, f"Expected 4 dimensions, got {env.action_space.shape}"
    assert env.action_space.high[0] == 1, f"Offload decision high should be 1"
    assert env.action_space.high[3] == 3, f"Bandwidth weight high should be 3"

    # Test sampling
    sample = env.action_space.sample()
    print(f"Sample action: {sample}")
    assert len(sample) == 4, "Sample should have 4 dimensions"
    assert 0 <= sample[0] <= 1, "Offload decision out of range"
    assert 0 <= sample[3] <= 3, "Bandwidth weight out of range"

    print("Action space test passed")

def test_dwdna_allocation():
    """Test DW-DNA bandwidth allocation"""
    env = EnvCore()

    print("\n=== Testing DW-DNA Allocation ===")

    # Test case 1: Single offloading UE
    weights = [3, 0, 0, 0, 0]
    offloads = [1, 0, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Single UE (weight=3): {bandwidths}")
    assert bandwidths[0] > 0, "Single UE should get bandwidth"
    assert sum(bandwidths[1:]) == 0, "Other UEs should get 0"

    # Test case 2: Equal weights
    weights = [2, 2, 2, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Three UEs equal weights: {bandwidths}")
    assert bandwidths[0] > 0 and bandwidths[1] > 0 and bandwidths[2] > 0
    assert abs(bandwidths[0] - bandwidths[1]) < env.kHz * 180 * 2  # Within 2 RB tolerance

    # Test case 3: Mixed weights
    weights = [3, 1, 0, 0, 0]
    offloads = [1, 1, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Mixed weights (3:1): {bandwidths}")
    assert bandwidths[0] > bandwidths[1], "Higher weight should get more bandwidth"

    print("DW-DNA allocation test passed")

if __name__ == "__main__":
    try:
        test_action_space()
        test_dwdna_allocation()
        print("\nIntegration tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)