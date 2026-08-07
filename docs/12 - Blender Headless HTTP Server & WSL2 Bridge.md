# 12 - Blender Headless HTTP Server & WSL2 Bridge

## Overview
Implemented a robust, cross-platform HTTP server bridge running natively inside Blender (`server.py`) on the Windows host. This server receives dynamic rendering triggers, orbital coordinates, and frame indices from an automated client loop (`render_loop.py`) executing inside WSL2.

---

## Architecture & Communication Flow

1. **Client (WSL2 Loop):** Automatically extracts the Windows host IP from the WSL2 routing table (`ip route show`), calculates target orbital parameters around the object, and sends HTTP POST requests to `http://<Host_IP>:12347`.
2. **Server (Blender Python):** Runs a custom `http.server.BaseHTTPRequestHandler` script directly via Blender's background command-line interface (`-b`).
3. **Execution & Auto-Targeting:** Intercepts request payloads, locks onto target objects (e.g., `Glass_Tube`) using strict datablock queries, computes target-relative camera positions, and executes Blender's native rendering operators synchronously.

---

## Implementation Details

### 1. Server Script (`server.py`)
* **Transport:** Python `http.server` bound to port `12347` on `0.0.0.0`.
* **Optics & Safety:** Enforces standard 50mm perspective optics and a 1mm near-clipping plane (`clip_start = 0.001`) to prevent geometry culling at close macro ranges.
* **Validation:** Writes output frames synchronously to disk and verifies persistence before returning a `200 OK` HTTP status code to the WSL2 client.

### 2. Client Loop (`render_loop.py`)
* **Dynamic Interop:** Uses a shell subprocess call to parse the Windows host gateway dynamically on startup.
* **Orbital Trajectory:** Computes smooth 360-degree circular trajectories at safe, repeatable radii around target meshes.