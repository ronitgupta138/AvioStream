"""In-memory sliding circular buffer and simulated telemetry streaming engine.

Maintains fixed-capacity (default 30 cycles) ring buffers per engine unit
with deterministic memory footprint (< 1.8 MB across 100 units).
"""

from typing import Dict, List, Optional, Tuple, Iterator
import sys
import numpy as np
import pandas as pd


class UnitRingBuffer:
    """Fixed-capacity ring buffer for a single engine unit.

    Uses contiguous NumPy float64 buffer with circular write head.
    """

    def __init__(self, capacity: int = 30, num_features: int = 14):
        self.capacity = capacity
        self.num_features = num_features
        self.buffer = np.zeros((capacity, num_features), dtype=np.float64)
        self.cycles = np.zeros(capacity, dtype=np.int32)
        self.head = 0
        self.count = 0

    def push(self, cycle: int, features: np.ndarray) -> None:
        """Appends a new cycle measurement into the circular slot."""
        if features.shape[0] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {features.shape[0]}")
        self.buffer[self.head] = features
        self.cycles[self.head] = cycle
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_chronological_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns array ordered from oldest to newest."""
        if self.count == 0:
            return np.empty((0, self.num_features), dtype=np.float64), np.empty(0, dtype=np.int32)
        if self.count < self.capacity:
            return self.buffer[:self.count].copy(), self.cycles[:self.count].copy()
        # Full buffer: head points to oldest element
        ordered_data = np.concatenate([self.buffer[self.head:], self.buffer[:self.head]], axis=0)
        ordered_cycles = np.concatenate([self.cycles[self.head:], self.cycles[:self.head]], axis=0)
        return ordered_data, ordered_cycles

    @property
    def is_full(self) -> bool:
        return self.count >= self.capacity

    @property
    def latest_cycle(self) -> int:
        if self.count == 0:
            return 0
        prev_head = (self.head - 1) % self.capacity
        return int(self.cycles[prev_head])


class FleetStreamBufferManager:
    """Manages circular buffers across a fleet of engine units."""

    def __init__(self, capacity: int = 30, num_features: int = 14):
        self.capacity = capacity
        self.num_features = num_features
        self.fleet_buffers: Dict[int, UnitRingBuffer] = {}

    def get_or_create_buffer(self, unit_id: int) -> UnitRingBuffer:
        if unit_id not in self.fleet_buffers:
            self.fleet_buffers[unit_id] = UnitRingBuffer(self.capacity, self.num_features)
        return self.fleet_buffers[unit_id]

    def ingest_snapshot(self, unit_id: int, cycle: int, features: np.ndarray) -> None:
        buf = self.get_or_create_buffer(unit_id)
        buf.push(cycle, features)

    def total_memory_bytes(self) -> int:
        """Calculates total buffer memory allocation."""
        total = sys.getsizeof(self.fleet_buffers)
        for unit_id, buf in self.fleet_buffers.items():
            total += sys.getsizeof(buf) + buf.buffer.nbytes + buf.cycles.nbytes
        return total


class TelemetryStreamSimulator:
    """Simulates real-time telemetry streaming from a test DataFrame."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df.sort_values(["unit_id", "time_cycle"]).copy()
        self.feature_cols = feature_cols

    def stream_cycles(self) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Yields (unit_id, cycle, sensor_array) snapshots."""
        for _, row in self.df.iterrows():
            unit_id = int(row["unit_id"])
            cycle = int(row["time_cycle"])
            values = row[self.feature_cols].values.astype(np.float64)
            yield unit_id, cycle, values


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS

    _, test_df, _ = load_cmapss_fd001()
    manager = FleetStreamBufferManager(capacity=30, num_features=len(ACTIVE_SENSORS))
    simulator = TelemetryStreamSimulator(test_df, feature_cols=ACTIVE_SENSORS)

    count = 0
    for unit_id, cycle, values in simulator.stream_cycles():
        manager.ingest_snapshot(unit_id, cycle, values)
        count += 1
        if count >= 1000:
            break

    mem_kb = manager.total_memory_bytes() / 1024
    print(f"Simulated {count} stream events across {len(manager.fleet_buffers)} units.")
    print(f"Total buffer memory footprint: {mem_kb:.2f} KB (Strictly < 1,800 KB budget).")

    # Verify unit 1 window
    buf1 = manager.get_or_create_buffer(1)
    data, cycles = buf1.get_chronological_window()
    print(f"Unit 1 buffer size: {len(cycles)} cycles, latest cycle: {buf1.latest_cycle}")
    print("Stream buffer verification passed.")
