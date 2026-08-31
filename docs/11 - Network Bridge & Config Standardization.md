

# 11 - Network Bridge & Config Standardization

## 1. Overview & Architecture
This milestone establishes cross-OS Inter-Process Communication (IPC) between our headless **WSL2 (Linux) ROS 2 environment** and our graphical **Blender 4.3 (Windows) digital twin**.

Instead of low-level sockets or subprocess execution, the `Label_Factory` pipeline communicates with Blender via an **HTTP REST API** hosted automatically by the Blender Annotation Tool (BAT) add-on over local network port `12345`.

```

[WSL2 / Linux Terminal] --(HTTP GET / POST via 172.24.192.1:12345)--> [Windows Host / Blender 4.3]

````

---

## 2. Configuration & Legacy Sanitization (`config.yaml`)
The baseline `config.yaml` file contained legacy hardcoded paths and mechanical tool references from the original developer (`karolyarmin`). These were purged and standardized for the medical robotics domain under user `dani`.

### Path Correction
* **Legacy:** `/home/karolyarmin/Label_Factory/...`
* **Updated:** `/home/dani/Projects/UR_Robotics_Project/Label_Factory/...`

### Object Registry Mapping
To allow automation scripts to push and pull $(X,Y,Z)$ coordinates without throwing `FileNotFoundError` or key errors, the 5 target classes were mapped to their corresponding Blender mesh names:

| Old Mechanical Key | New Medical Key | Blender Mesh Name |
| :--- | :--- | :--- |
| `object_pose_Cube` | `object_pose_Glass_Tube` | `Glass_Tube` |
| `object_pose_Plier_Long` | `object_pose_Glass_Slide` | `Glass_Slide` |
| `object_pose_Plier` | `object_pose_Slide_Holder` | `Slide_Holder` |
| `object_pose_Rachet` | `object_pose_Tube_Holder` | `Tube_Holder` |
| `object_pose_Screw_driver` | `object_pose_Cap` | `Cap` |

---

## 3. WSL2 vs. Windows Networking Fix
By default, WSL2 operates inside a lightweight Hyper-V virtual machine with its own virtual subnet.

* **The Problem:** Automation scripts in `BAT_scene_opt.py` were hardcoded to `http://localhost:12345`, causing `ConnectionRefusedError` because Linux attempted to connect to its own internal loopback interface rather than the Windows host.
* **The Resolution:** 
  1. Identified the Windows host IP from WSL2 routing tables:
     ```bash
     ip route show | grep -i default | awk '{ print $3}'
     # Output: 172.24.192.1
     ```
  2. Patched `BAT_scene_opt.py` endpoints to target `http://172.24.192.1:12345`.

---

## 4. Naming Convention Standardization (Underscores vs. Spaces)
* **The Problem:** HTTP REST URLs encode spaces as `%20`. While querying `/object?name=Glass%20Tube` succeeded manually, automated Python/ROS2 string-matching scripts expect 1:1 parity between `config.yaml` keys and Blender object names.
* **The Resolution:** Replaced spaces with underscores across all target meshes in the Blender Outliner (`Glass_Tube`, `Glass_Slide`, `Slide_Holder`, `Tube_Holder`, `Cap`) and verified zero-error string matching.

---

## 5. Verification Proof
Executed an end-to-end HTTP `GET` query from WSL2 Linux to Windows Blender:
```bash
curl -i "[http://172.24.192.1:12345/object?name=Glass_Tube](http://172.24.192.1:12345/object?name=Glass_Tube)"
````

**Response Received (`200 OK`):**

JSON

```
{
  "status": "success",
  "object": "Glass_Tube",
  "location": [0.2728855609893799, 9.205709457397461, 3.623335361480713],
  "rotation": [0.0, 0.0, 0.0]
}
```