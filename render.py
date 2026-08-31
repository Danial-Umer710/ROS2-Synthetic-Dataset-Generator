import bpy
import json
import math
import os
import sys
import http.server
from mathutils import Vector

PORT = 12347
OUTPUT_DIR = r"D:\Projects\UR_Robotics_Project\Test Renders"

def log(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

def get_target_tube():
    """Strictly targets Glass_Tube first to avoid accidentally locking onto Tube_Holder."""
    exact_tube = bpy.data.objects.get("Glass_Tube")
    if exact_tube:
        return exact_tube
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name not in ["Tube_Holder", "Ground", "Cube", "Glass_Slide"]:
            return obj
    return None

class ExternalRenderHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        log("\n>>> EXTERNAL HANDLER HIT <<<")
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8')) if body else {}
            frame_idx = data.get("frame_index", 0)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = os.path.join(OUTPUT_DIR, f"{frame_idx:04d}.png")
            log(f"Target Render Path: {filename}")

            # 1. Verify & Auto-Assign Active Camera
            scene = bpy.context.scene
            if not scene.camera:
                for obj in bpy.data.objects:
                    if obj.type == 'CAMERA':
                        scene.camera = obj
                        log(f"-> Automatically linked Camera: {obj.name}")
                        break

            if not scene.camera:
                raise RuntimeError("Render failed: No object of type 'CAMERA' exists in Models.blend.")

            # 2. Lock Standard Robotics Optics (Perspective | 50mm Lens | 1mm Clip Start)
            scene.camera.data.type = 'PERSP'
            scene.camera.data.clip_start = 0.001
            scene.camera.data.lens = 50.0

            # 3. Locate Target Object (Strictly Glass_Tube)
            scene.frame_set(frame_idx)
            target_obj = get_target_tube()
            
            if target_obj:
                target_pos = target_obj.location
                log(f"Target Locked -> '{target_obj.name}' at World Coords: ({target_pos.x:.2f}, {target_pos.y:.2f}, {target_pos.z:.2f})")
            else:
                target_pos = Vector((0.0, 0.0, 0.0))
                log("CRITICAL WARNING: 'Glass_Tube' not found! Defaulting to (0,0,0).")

            # 4. Deterministic Orbit Coordinates (No Auto-Guessing)
            angle = data.get("orbit_angle", 0.0)
            radius = data.get("orbit_radius", 0.12)
            height = data.get("orbit_height", 0.06)

            cam_x = target_pos.x + (radius * math.cos(angle))
            cam_y = target_pos.y + (radius * math.sin(angle))
            cam_z = target_pos.z + height
            scene.camera.location = Vector((cam_x, cam_y, cam_z))

            # 5. Auto-Aim Camera Lens (-Z axis) Directly at Glass_Tube
            direction = target_pos - scene.camera.location
            if direction.length > 0.0001:
                scene.camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            log(f"Camera positioned at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) aiming at {target_obj.name if target_obj else 'Origin'}.")

            # 6. Configure & Execute Render
            scene.render.filepath = filename
            scene.render.image_settings.file_format = 'PNG'
            
            log("Executing Blender Render Operator...")
            result = bpy.ops.render.render(write_still=True)
            log(f"Render Operator Result: {result}")

            # 7. Strict Disk Verification
            if 'FINISHED' in result and os.path.exists(filename):
                log(f"SUCCESS: Confirmed file on disk -> {filename}")
                self.send_response(200)
            else:
                log(f"FAILURE: Blender returned {result} or file was not created.")
                self.send_response(500)

            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        except Exception as e:
            log(f"ERROR IN HANDLER: {e}")
            self.send_response(500)
            self.end_headers()

if __name__ == '__main__':
    server = http.server.HTTPServer(('0.0.0.0', PORT), ExternalRenderHandler)
    log(f"*** TRUE EXTERNAL HTTP SERVER LIVE ON PORT {PORT} ***")
    server.serve_forever()
