import bpy
import json
import math
import os
import sys
import random
import http.server
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

PORT = 12348
OUTPUT_DIR = r"D:\Projects\UR_Robotics_Project\Test Renders"

ORIGINAL_Z = None
ORIGINAL_PITCH = None
ORIGINAL_ROLL = None

def log(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

def setup_cycles_engine(scene):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    try:
        cprefs = prefs.addons['cycles'].preferences
        cprefs.compute_device_type = 'CUDA'
    except Exception:
        pass
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True 

def isolate_target_object(target_obj):
    """NUCLEAR OPTION: Physically teleport unwanted meshes underground."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Whitelist: Leave the target and the patterned floor alone
            if obj == target_obj or obj.name == "Charuco":
                obj.hide_render = False
            else:
                # Blacklist: Shove the giant pillars and ground 100 meters straight down
                obj.location[2] = -100.0
                obj.hide_render = True

def get_target_tube():
    exact_tube = bpy.data.objects.get("Glass_Tube")
    if exact_tube:
        return exact_tube
    return None

def randomize_environment(target_obj):
    global ORIGINAL_Z, ORIGINAL_PITCH, ORIGINAL_ROLL
    
    if ORIGINAL_Z is None:
        ORIGINAL_Z = target_obj.location.z
        ORIGINAL_PITCH = target_obj.rotation_euler[0]
        ORIGINAL_ROLL = target_obj.rotation_euler[1]

    # Position scattering
    target_obj.location.x = random.uniform(-0.08, 0.08)
    target_obj.location.y = random.uniform(-0.08, 0.08)
    target_obj.location.z = ORIGINAL_Z
    target_obj.rotation_euler[0] = ORIGINAL_PITCH  
    target_obj.rotation_euler[1] = ORIGINAL_ROLL  
    target_obj.rotation_euler[2] = random.uniform(0.0, 2.0 * math.pi)

    # Lighting intensity & angle variation
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            if obj.data.type == 'SUN':
                obj.data.energy = random.uniform(3.0, 10.0)  
                obj.rotation_euler = (random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4), random.uniform(0.0, 2.0 * math.pi)) 
            else:
                obj.data.energy = 3000.0  
                obj.location = (target_obj.location.x, target_obj.location.y, 0.5)
    
    bpy.context.view_layer.update()

def get_yolo_bbox(obj, scene, camera):
    bpy.context.view_layer.update()
    
    mat = obj.matrix_world
    bbox_corners = [mat @ Vector(corner) for corner in obj.bound_box]
    projected = [world_to_camera_view(scene, camera, coord) for coord in bbox_corners]
    
    xs = [p.x for p in projected]
    ys = [p.y for p in projected]
    
    min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
    min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
    
    width = max_x - min_x
    height = max_y - min_y
    
    x_center = min_x + (width / 2.0)
    y_center = 1.0 - (min_y + (height / 2.0))
    
    return [0, x_center, y_center, width, height]

class ExternalRenderHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8')) if body else {}
            frame_idx = data.get("frame_index", 0)
            do_randomize = data.get("randomize", False)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            img_path = os.path.join(OUTPUT_DIR, f"{frame_idx:04d}.png")
            txt_path = os.path.join(OUTPUT_DIR, f"{frame_idx:04d}.txt")

            scene = bpy.context.scene
            setup_cycles_engine(scene)

            if not scene.camera:
                for obj in bpy.data.objects:
                    if obj.type == 'CAMERA':
                        scene.camera = obj
                        break

            if not scene.camera:
                raise RuntimeError("No active camera found.")

            scene.frame_set(frame_idx)
            target_obj = get_target_tube()
            
            if target_obj:
                # Force the teleportation
                isolate_target_object(target_obj)
                
                if do_randomize:
                    randomize_environment(target_obj)

            target_pos = target_obj.location if target_obj else Vector((0.0, 0.0, 0.0))
            angle = data.get("orbit_angle", 0.0)
            radius = data.get("orbit_radius", 0.15)
            height = data.get("orbit_height", 0.08)

            cam_x = target_pos.x + (radius * math.cos(angle))
            cam_y = target_pos.y + (radius * math.sin(angle))
            cam_z = target_pos.z + height
            scene.camera.location = Vector((cam_x, cam_y, cam_z))

            direction = target_pos - scene.camera.location
            if direction.length > 0.0001:
                scene.camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            scene.render.filepath = img_path
            scene.render.image_settings.file_format = 'PNG'
            result = bpy.ops.render.render(write_still=True)

            if target_obj and 'FINISHED' in result:
                bbox = get_yolo_bbox(target_obj, scene, scene.camera)
                with open(txt_path, "w") as f:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}")
                log(f"SUCCESS: Saved {img_path}")
            else:
                log("FAILURE: Render failed.")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        except Exception as e:
            log(f"ERROR: {e}")
            self.send_response(500)
            self.end_headers()

if __name__ == '__main__':
    server = http.server.HTTPServer(('0.0.0.0', PORT), ExternalRenderHandler)
    log(f"*** EXTERNAL HTTP SERVER LIVE ON PORT {PORT} ***")
    server.serve_forever()
