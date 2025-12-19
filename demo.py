import argparse
import json
import os
import shutil
import sys
import math
import pandas as pd
import numpy as np
import tkinter as tk
import threading
from isaacsim import SimulationApp
import carb

# ---------------------------------------------------------
# 1. EVALUATION & AESTHETIC UI FUNCTIONS
# ---------------------------------------------------------

def calc_ndtw(pred_traj, gt_traj, threshold=3.0):
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return 0.0
    pred_xy = np.array(pred_traj)
    gt_xy = np.array(gt_traj)
    N, M = len(pred_xy), len(gt_xy)
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1])
            dtw[i, j] = dist + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dtw_distance = dtw[N, M]
    ndtw = np.exp(-dtw_distance / (threshold * len(gt_xy)))
    return float(ndtw)

def calculate_trajectory_length(coords):
    if len(coords) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(coords)):
        dist = math.sqrt((coords[i][0] - coords[i-1][0])**2 + 
                        (coords[i][1] - coords[i-1][1])**2)
        total_length += dist
    return total_length

def evaluate_single_episode(csv_file, episode_data):
    """
    Returns both a string (for console) and a dict (for the UI).
    """
    if not os.path.exists(csv_file):
        print("Error: CSV file not found.")
        return None, None

    df = pd.read_csv(csv_file)
    coords = df[['pos_x', 'pos_y']].values.tolist()

    filtered_coords = [coords[0]]
    for point in coords[1:]:
        if point != filtered_coords[-1]:
            filtered_coords.append(point)

    goal = episode_data['goals']['position']
    goal_x, goal_y = goal[0], goal[1]
    
    reference_path = episode_data['reference_path']
    gt_coords = [[pos[0], pos[1]] for pos in reference_path]

    last_x, last_y = filtered_coords[-1]
    dist_last = math.sqrt((goal_x - last_x)**2 + (goal_y - last_y)**2)
    sr = 1 if dist_last <= 3 else 0

    osr = 0
    for idx, (x, y) in enumerate(filtered_coords):
        if math.sqrt((goal_x - x)**2 + (goal_y - y)**2) <= 3:
            osr = 1
            break

    ndtw_score = calc_ndtw(filtered_coords, gt_coords, threshold=3.0)
    tl = calculate_trajectory_length(filtered_coords)
    tl_gt = calculate_trajectory_length(gt_coords)
    
    if tl_gt > 0 and tl > 0:
        spl = sr * (tl_gt / max(tl, tl_gt))
    else:
        spl = 0.0

    # Console Output
    result_str = (
        f"--- Episode Evaluation ---\n"
        f"SR: {sr}, OSR: {osr}, SPL: {spl:.3f}, nDTW: {ndtw_score:.3f}, TL: {tl:.2f}m"
    )
    print(f"\n{result_str}\n")
    
    # Dictionary for GUI
    metrics = {
        "SR": sr,
        "OSR": osr,
        "SPL": spl,
        "nDTW": ndtw_score,
        "TL": tl,
        "Goal Dist": dist_last
    }
    return result_str, metrics

def show_beautiful_popup(metrics):
    """
    Displays a styled, modern popup for results.
    """
    if metrics is None:
        return

    root = tk.Tk()
    root.title("Evaluation Results")
    
    # Calculate screen center
    w, h = 400, 450
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    root.configure(bg="#f0f2f5")
    root.attributes('-topmost', True)
    root.resizable(False, False)

    # -- Header Section --
    is_success = metrics["SR"] == 1
    header_color = "#2ecc71" if is_success else "#e74c3c" # Green or Red
    header_text = "MISSION SUCCESS!" if is_success else "MISSION FAILED"
    
    header_frame = tk.Frame(root, bg=header_color, height=80)
    header_frame.pack(fill='x')
    
    lbl_header = tk.Label(
        header_frame, 
        text=header_text, 
        font=("Helvetica", 18, "bold"), 
        bg=header_color, 
        fg="white"
    )
    lbl_header.pack(pady=20)

    # -- Content Section --
    content_frame = tk.Frame(root, bg="white", padx=20, pady=20)
    content_frame.pack(fill='both', expand=True, padx=15, pady=15)

    # Helper to create rows
    def create_row(parent, label, value, row_idx):
        lbl = tk.Label(parent, text=label, font=("Arial", 11, "bold"), bg="white", fg="#555")
        lbl.grid(row=row_idx, column=0, sticky="w", pady=8)
        
        val_txt = f"{value:.3f}" if isinstance(value, float) else str(value)
        val = tk.Label(parent, text=val_txt, font=("Consolas", 12), bg="white", fg="#333")
        val.grid(row=row_idx, column=1, sticky="e", pady=8)

    # Grid config
    content_frame.columnconfigure(0, weight=1)
    content_frame.columnconfigure(1, weight=1)

    create_row(content_frame, "Success Rate (SR):", metrics["SR"], 0)
    create_row(content_frame, "Oracle Success (OSR):", metrics["OSR"], 1)
    create_row(content_frame, "SPL Score:", metrics["SPL"], 2)
    create_row(content_frame, "nDTW Score:", metrics["nDTW"], 3)
    create_row(content_frame, "Trajectory Length:", f"{metrics['TL']:.2f} m", 4)
    create_row(content_frame, "Dist to Goal:", f"{metrics['Goal Dist']:.2f} m", 5)

    # -- Footer / Button --
    btn = tk.Button(
        root, 
        text="Close & Finish", 
        command=root.destroy, 
        bg="#34495e", 
        fg="white", 
        font=("Arial", 11, "bold"),
        relief="flat",
        height=2
    )
    btn.pack(fill='x', padx=15, pady=(0, 15))

    root.mainloop()

def show_instruction_left_centered(instruction_text):
    """Shows instruction window vertically centered on the left edge."""
    def create_window():
        root = tk.Tk()
        root.title("Task")
        
        # Window Dimensions
        win_w, win_h = 300, 650
        
        # Get Screen Dimensions
        screen_h = root.winfo_screenheight()
        
        # Calculate Y to be in the middle
        pos_y = (screen_h - win_h) // 2
        pos_x = 0  # Left edge
        
        root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        root.attributes('-topmost', True)
        root.resizable(False, False)
        
        # Styling
        label = tk.Label(root, text=instruction_text, wraplength=280, justify='left',
            padx=15, pady=20, font=("Helvetica", 12, "bold"), fg="#2c3e50", bg="#f9f9f9")
        label.pack(expand=True, fill='both')
        
        hint_frame = tk.Frame(root, bg="#f9f9f9")
        hint_frame.pack(fill='x', pady=10)
        
        tk.Label(hint_frame, text="Press", font=("Arial", 10), bg="#f9f9f9", fg="#7f8c8d").pack(side="top")
        tk.Label(hint_frame, text="[ ENTER ]", font=("Courier", 12, "bold"), bg="#f9f9f9", fg="#e74c3c").pack(side="top")
        tk.Label(hint_frame, text="to finish recording", font=("Arial", 10), bg="#f9f9f9", fg="#7f8c8d").pack(side="top")

        btn = tk.Button(root, text="Minimize", command=root.iconify, font=("Arial", 9))
        btn.pack(pady=10)
        
        root.focus_force()
        root.lift()
        root.mainloop()
    
    thread = threading.Thread(target=create_window, daemon=True)
    thread.start()

# ---------------------------------------------------------
# 2. ISAAC SIM SETUP
# ---------------------------------------------------------

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",  
        "headless": True,
    },
    "resolution": [512, 512],
    "writer": "BasicWriter",
}

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, help="0-56")
parser.add_argument("--task", default="fine", help="fine | coarse")
parser.add_argument("--work_dir", default="/data/lsh/isaac_code/demo")
parser.add_argument("--headless", default=False, action="store_true")
parser.add_argument("--mode", default="manual", help="manual | teleop | policy")
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--record", default=True, action="store_true")
parser.add_argument("--camera_mode", default="floating")  
parser.add_argument("--camera_height", default=1.5, type=float, help="Camera height from ground")
args, unknown_args = parser.parse_known_args()

config["launch_config"]["headless"] = args.headless
simulation_app = SimulationApp(config["launch_config"])

def rotation_from_direction(direction, up_vector=np.array([0, 0, 1])):
    from scipy.spatial.transform import Rotation as R
    direction = np.array(direction, dtype=np.float64)
    forward = direction / np.linalg.norm(direction)
    right = np.cross(up_vector, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    up = np.cross(forward, right)
    rot_mat = np.column_stack((forward, right, up))
    quat = R.from_matrix(rot_mat).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]])


class FloatingCameraController:
    def __init__(self, world, scene_data):
        self.world = world
        self.state = 0
        from pxr import UsdGeom

        self.camera_height = args.camera_height 
        self.start_position = np.array(scene_data["start_location"]) + np.array([0, 0, self.camera_height])
        self.start_orientation = np.array(scene_data["start_orientation"])

        stage = simulation_app.context.get_stage()
        self.camera_path = "/World/FloatingCamera"
        
        camera_prim = UsdGeom.Camera.Define(stage, self.camera_path)
        camera_prim.CreateFocalLengthAttr(10.0)
        camera_prim.CreateClippingRangeAttr().Set((0.01, 1000.0))
        camera_prim.CreateVerticalApertureAttr(20)
        camera_prim.CreateHorizontalApertureAttr(20.0)

        self.camera = self.camera_path 
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.look_direction = np.array([1, 0, 0]) 

        self.move_speed = 0.25  
        self.turn_speed = 15.0  

        self.traj = scene_data.get("traj", None)
        self.traj_index = 0
        self.traj_dir = 1
        self.mode = args.mode
        self._base_command = np.array([0.0, 0.0, 0.0])
        self.mission_complete = False 

    def reset(self):
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.look_direction = np.array([1, 0, 0])
        self._update_camera()
        self.state = 0
        self.traj_index = 0
        self.traj_dir = 1
        self.mission_complete = False
        print('=' * 10, "reset", "=" * 10)

    def init_manual(self):
        import omni.appwindow
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._base_command = np.array([0.0, 0.0, 0.0])
        pos_del = 1
        self._input_keyboard_mapping = {
            "W": [pos_del, 0., 0.],
            "A": [0., 0., pos_del],
            "D": [0., 0., -pos_del],
            "S": [-pos_del, 0., 0.]
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Check for ENTER key to Finish Mission
            if event.input.name == "ENTER":
                print("ENTER pressed - Mission Complete.")
                self.mission_complete = True
                self.state = "Done"
                return True
                
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _update_camera(self):
        from pxr import UsdGeom, Gf
        stage = simulation_app.context.get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)

        if camera_prim and camera_prim.IsValid():
            xformable = UsdGeom.Xformable(camera_prim)
            xformable.ClearXformOpOrder()
            
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(self.current_position[0],
                                     self.current_position[1],
                                     self.current_position[2]))

            orientation = rotation_from_direction(self.look_direction)
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])  # xyzw
            euler = r.as_euler('xyz', degrees=True)
            rotate_op = xformable.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(euler[0] + 90, euler[1], euler[2] - 90)) 

    def get_camera_transform(self):
        return {
            "pos_x": self.current_position[0],
            "pos_y": self.current_position[1],
            "pos_z": self.current_position[2],
            "look_x": self.look_direction[0],
            "look_y": self.look_direction[1],
            "look_z": self.look_direction[2],
        }

    def run(self, step_size):
        from scipy.spatial.transform import Rotation as R
        if self.mode == 'manual':
            if abs(self._base_command[0]) > 0:
                self.current_position += self.look_direction * self._base_command[0] * self.move_speed * step_size
            if abs(self._base_command[1]) > 0:
                right_dir = np.cross(self.look_direction, np.array([0, 0, 1]))
                right_dir = right_dir / np.linalg.norm(right_dir)
                self.current_position += right_dir * self._base_command[1] * self.move_speed * step_size
            if abs(self._base_command[2]) > 0:
                angle = self._base_command[2] * self.turn_speed * step_size
                r = R.from_euler('z', angle, degrees=True)
                self.look_direction = r.apply(self.look_direction)
            self.current_position[2] = self.camera_height
        self._update_camera()

reset_needed = False
first_step = True

def run(scene_data, output_dir=None, episode_id=None):
    from pxr import Sdf
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    import omni.replicator.core as rep

    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=8.0 / 200.0)
    my_world.scene.add_default_ground_plane(z_position=0, name="default_ground_plane", prim_path="/World/defaultGroundPlane")

    if "usd_path" in scene_data:
        prim = define_prim("/World/Ground", "Xform")
        asset_path = scene_data["usd_path"]
        prim.GetReferences().AddReference(asset_path, "/Root")

    camera_controller = FloatingCameraController(world=my_world, scene_data=scene_data)
    if args.mode == "manual":
        camera_controller.init_manual()

    stage = simulation_app.context.get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    my_world.reset()
    camera_controller.reset()

    global reset_needed
    def on_physics_step(step_size):
        global first_step, reset_needed
        if first_step:
            camera_controller.reset()
            first_step = False
        elif reset_needed:
            my_world.reset(True)
            reset_needed = False
            first_step = True
        else:
            camera_controller.run(step_size)

    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)
    
    camera_transforms = []
    frame = 0
    mission_complete_signal = False

    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_stopped() and not reset_needed: reset_needed = True
        
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                camera_controller.reset()
                frame = 0
                reset_needed = False
                camera_transforms = []

            if args.render:
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            
            if args.record and (frame % 50 == 0):
                transform = camera_controller.get_camera_transform()
                transform["frame"] = frame
                camera_transforms.append(transform)

            if camera_controller.mission_complete:
                mission_complete_signal = True
                break

            if args.headless and frame > 2000:
                break
            frame += 1

    if not mission_complete_signal:
        print("\n\n [ABORT] Simulation window closed without pressing ENTER.")
        return None, False

    if args.record:
        print("save camera_transforms", len(camera_transforms))
        df = pd.DataFrame(camera_transforms)
        csv_save_path = os.path.join(output_dir, f"{episode_id}.csv")
        df.to_csv(csv_save_path, index=False)
        return csv_save_path, True
    
    return None, True

if __name__ == '__main__':
    work_dir = os.getcwd()

    task = args.task
    print(task)

    with open(f"{task}_grained_demo.json", 'r') as f:
        data = json.load(f)

    cur_episode = data[int(args.index)]
    scene_id = cur_episode['scan']
    loc = cur_episode['start_position']
    ori = cur_episode['start_rotation']
    instruction_text = cur_episode['instruction']['instruction_text']

    if task == "coarse":
        instruction_text = instruction_text["natural"]
    
    # 1. SHOW INSTRUCTION (Left Center)
    show_instruction_left_centered(instruction_text)
    usd_path = os.path.join(work_dir, scene_id, f'{scene_id}.usda')
    
    traj = []
    for i in range(10):
        traj.append([loc[0] + i * 0.5, loc[1], loc[2]])
    loc[-1] = 1.5
    scene_data = {
        "usd_path": usd_path,
        "start_location": loc,
        "start_orientation": ori,
        "traj": np.array(traj),
    }
    
    output_dir = args.work_dir
    print(f"Instruction: {instruction_text}")
    print(" >>> PRESS [ENTER] TO FINISH RECORDING <<<")
    
    # 2. RUN SIM
    saved_csv_path, success = run(scene_data, output_dir, cur_episode['episode_id'])
    
    # simulation_app.close()

    if not success:
        sys.exit(0)

    # 3. EVALUATE & SHOW BEAUTIFUL POPUP
    if saved_csv_path and os.path.exists(saved_csv_path):
        print("\nEvaluating Performance...")
        # Get metrics dict
        _, metrics = evaluate_single_episode(saved_csv_path, cur_episode)
        
        # Show Beautiful GUI
        show_beautiful_popup(metrics)
    
    print(f"Done.")