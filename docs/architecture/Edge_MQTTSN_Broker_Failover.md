# Edge MQTT-SN Broker Failover & High-Availability Ingestion Topology

## 1. Architecture
Airborne telemetry forwarders publish compact telemetry messages over MQTT for Sensor Networks (MQTT-SN) via low-bandwidth satellite links (ACARS / Inmarsat).

## 2. Local Volatile Ring Buffering
When satellite connectivity drops:
- Packets are buffered into an in-memory lock-free circular ring buffer (capacity: 250,000 packets).
- **Eviction Policy:** High-priority health alarms are retained; high-frequency ambient air temperature ticks are downsampled.

## 3. Reconnection & Burst Flush
Upon satellite link restoration, the edge client initiates burst chunk transmission with TCP window scaling, preventing packet loss.
