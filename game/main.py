import cv2
import pygame
import threading
import queue
import random 
from ultralytics import YOLO
import time
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
FPS = 60

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

MODEL_PATH = 'yolov8n-pose.engine' 
model = YOLO(MODEL_PATH)

def open_camera():
    """負責處理 Jetson 平台上的攝影機初始化"""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if cap.isOpened():
        print("[INFO] 成功開啟 USB 攝影機 (/dev/video0)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        return cap
        
    print("[WARN] 無法開啟 USB 攝影機，嘗試開啟 CSI 排線攝影機...")
    # USB 無法開啟時，改用 Jetson CSI 攝影機管線。
    gstreamer_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)NV12, framerate=(fraction)30/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("[INFO] 成功開啟 CSI 攝影機 (GStreamer)")
    else:
        print("[ERROR] 無法找到任何可用的攝影機裝置！")
        
    return cap

def analyze_pose(keypoints, box, player_id, was_extended):
    result = {"attack": "idle", "move": "idle", "ready": False, "defend": False}
    if len(keypoints) < 13: return result, False

    l_sh, r_sh = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
    if l_sh[0] > 0 and r_sh[0] > 0:
        sh_x, sh_y = (l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2
    elif l_sh[0] > 0:
        sh_x, sh_y = l_sh[0], l_sh[1]
    elif r_sh[0] > 0:
        sh_x, sh_y = r_sh[0], r_sh[1]
    else:
        return result, False

    l_hip, r_hip = keypoints[L_HIP], keypoints[R_HIP]
    if l_hip[1] > 0 and r_hip[1] > 0:
        hip_y = (l_hip[1] + r_hip[1]) / 2
    else:
        hip_y = l_hip[1] if l_hip[1] > 0 else r_hip[1]
    
    torso = abs(hip_y - sh_y)
    if torso < 40: torso = 80 
    
    punch_dist = torso * 0.75  
    height_tolerance = torso * 0.8 

    # 優先判定防禦，避免雙手舉高時誤觸攻擊。
    if keypoints[L_WRIST][1] > 0 and keypoints[R_WRIST][1] > 0:
        l_wrist_y = keypoints[L_WRIST][1]
        r_wrist_y = keypoints[R_WRIST][1]
        defense_line = sh_y + (torso * 0.1)
        if l_wrist_y < defense_line and r_wrist_y < defense_line:
            result["defend"] = True
            return result, False 

    is_currently_extended = False
    for wr_idx in [L_WRIST, R_WRIST]:
        if keypoints[wr_idx][0] > 0:
            wr = keypoints[wr_idx]
            forward_extension = (wr[0] - sh_x) if player_id == 1 else (sh_x - wr[0])
            y_deviation = abs(wr[1] - sh_y) 
            
            if forward_extension > punch_dist and y_deviation < height_tolerance:
                if forward_extension > (y_deviation * 0.75):
                    is_currently_extended = True
                    break

    # 手臂由伸直轉為收回時觸發一次攻擊。
    if was_extended and not is_currently_extended:
        result["attack"] = "punch"

    # 以肩膀相對髖部的偏移判定傾斜，兩位玩家的前進方向相反。
    mid_shoulder_x = sh_x
    mid_hip_x = (l_hip[0] + r_hip[0]) / 2 if l_hip[0]>0 and r_hip[0]>0 else (l_hip[0] if l_hip[0]>0 else r_hip[0])
    lean_threshold = (box[2] - box[0]) * 0.16 
    
    if player_id == 1:
        if mid_shoulder_x > mid_hip_x + lean_threshold: result["move"] = "forward"
        elif mid_shoulder_x < mid_hip_x - lean_threshold: result["move"] = "backward"
    elif player_id == 2:
        if mid_shoulder_x < mid_hip_x - lean_threshold: result["move"] = "forward"
        elif mid_shoulder_x > mid_hip_x + lean_threshold: result["move"] = "backward"

    return result, is_currently_extended

def pose_estimation_worker(state_queue):
    cap = open_camera()
    p1_was_extended = False
    p2_was_extended = False
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
            
        frame = cv2.flip(frame, 1)
        
        # 計時涵蓋整個模型呼叫，包含前處理與後處理。
        t_start = time.time()
        
        results = model(frame, verbose=False)
        
        t_end = time.time()
        inference_speed = (t_end - t_start) * 1000

        game_state = {"1P": None, "2P": None, "inf_time": inference_speed}

        if len(results) > 0 and results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints = results[0].keypoints.xy.cpu().numpy()

            persons = []
            for i in range(len(boxes)):
                center_x = (boxes[i][0] + boxes[i][2]) / 2
                persons.append({"center_x": center_x, "box": boxes[i], "keypoints": keypoints[i]})
            
            persons = sorted(persons, key=lambda x: x["center_x"])
            midpoint = CAM_WIDTH / 2
            
            p1_detected = False
            p2_detected = False
            
            for p in persons:
                if p["center_x"] < midpoint and game_state["1P"] is None:
                    game_state["1P"], p1_was_extended = analyze_pose(p["keypoints"], p["box"], 1, p1_was_extended)
                    p1_detected = True
                elif p["center_x"] >= midpoint and game_state["2P"] is None:
                    game_state["2P"], p2_was_extended = analyze_pose(p["keypoints"], p["box"], 2, p2_was_extended)
                    p2_detected = True
            
            if not p1_detected: p1_was_extended = False
            if not p2_detected: p2_was_extended = False

        # 丟棄尚未消費的舊結果，避免辨識資料累積造成操作延遲。
        if not state_queue.empty():
            try: state_queue.get_nowait()
            except queue.Empty: pass
        state_queue.put(game_state)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pose Fighter - Clean White UI Edition")
    clock = pygame.time.Clock()

    pygame.font.init()
    debug_font = pygame.font.SysFont('consolas', 16)
    title_font = pygame.font.SysFont('arial', 32, bold=True)
    countdown_font = pygame.font.SysFont('impact', 140, bold=True)

    p1_hp, p2_hp = 100, 100
    player_w, player_h = 100, 150 
    player_y = 200

    p1_hp_buffer = 100
    p2_hp_buffer = 100

    hitstop_timer = 0
    shake_timer = 0
    shake_intensity = 0

    p1_combo = 0
    p1_combo_timer = 0
    p2_combo = 0
    p2_combo_timer = 0

    p1_wall_bounce_text = 0
    p2_wall_bounce_text = 0

    bg_img = pygame.image.load("background.png").convert()
    bg_img = pygame.transform.scale(bg_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
    p1_idle_img = pygame.transform.scale(pygame.image.load("player1_idle.png").convert_alpha(), (player_w, player_h))
    p1_punch_img = pygame.transform.scale(pygame.image.load("player1_punch.png").convert_alpha(), (player_w + 50, player_h))
    p1_defend_img = pygame.transform.scale(pygame.image.load("player1_defend.png").convert_alpha(), (player_w, player_h))
    p2_idle_img = pygame.transform.scale(pygame.image.load("player2_idle.png").convert_alpha(), (player_w, player_h))
    p2_punch_img = pygame.transform.scale(pygame.image.load("player2_punch.png").convert_alpha(), (player_w + 50, player_h))
    p2_defend_img = pygame.transform.scale(pygame.image.load("player2_defend.png").convert_alpha(), (player_w, player_h))
    
    state_queue = queue.Queue(maxsize=1)
    worker_thread = threading.Thread(target=pose_estimation_worker, args=(state_queue,), daemon=True)
    
    worker_thread.start()

    game_state_mode = "WAITING_START"
    countdown_frames = FPS * 4 
    game_over_timer = 0

    p1_state = {"pos_x": 150, "punch_timer": 0, "hit_spark_timer": 0, "guard_text_timer": 0, "stun_timer": 0, "energy": 0, "crit_text_timer": 0}
    p2_state = {"pos_x": SCREEN_WIDTH - 150, "punch_timer": 0, "hit_spark_timer": 0, "guard_text_timer": 0, "stun_timer": 0, "energy": 0, "crit_text_timer": 0}

    # 攻擊姿態維持固定幀數，避免隨辨識結果閃爍。
    PUNCH_HOLD_FRAMES = 15 
    EFFECT_DURATION = 8
    MOVE_SPEED = 4  
    MIN_DISTANCE = player_w - 20 

    p1_move, p1_is_defend = "idle", False
    p2_move, p2_is_defend = "idle", False

    current_state = {"1P": None, "2P": None}
    
    current_inf_time = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if p1_hp_buffer > p1_hp: p1_hp_buffer -= 0.4
        if p2_hp_buffer > p2_hp: p2_hp_buffer -= 0.4

        if p1_combo_timer > 0:
            p1_combo_timer -= 1
            if p1_combo_timer == 0: p1_combo = 0
        if p2_combo_timer > 0:
            p2_combo_timer -= 1
            if p2_combo_timer == 0: p2_combo = 0

        if hitstop_timer > 0:
            hitstop_timer -= 1
        else:
            if p1_state["stun_timer"] > 0: p1_state["stun_timer"] -= 1
            if p2_state["stun_timer"] > 0: p2_state["stun_timer"] -= 1
            if p1_state["crit_text_timer"] > 0: p1_state["crit_text_timer"] -= 1
            if p2_state["crit_text_timer"] > 0: p2_state["crit_text_timer"] -= 1

            p1_body_rect = pygame.Rect(p1_state["pos_x"] - player_w//2, player_y, player_w, player_h)
            p1_punch_rect = pygame.Rect(p1_state["pos_x"] - player_w//2, player_y, player_w + 50, player_h)
            p2_body_rect = pygame.Rect(p2_state["pos_x"] - player_w//2, player_y, player_w, player_h)
            p2_punch_rect = pygame.Rect(p2_state["pos_x"] - player_w//2 - 50, player_y, player_w + 50, player_h)

            if not state_queue.empty():
                current_state = state_queue.get()
                
                if "inf_time" in current_state:
                    current_inf_time = current_state["inf_time"]
                
                if current_state.get("1P") and p1_state["stun_timer"] == 0:
                    p1_move = current_state["1P"]["move"]
                    p1_is_defend = current_state["1P"]["defend"]
                    if current_state["1P"]["attack"] == "punch" and p1_state["punch_timer"] == 0:
                        p1_state["punch_timer"] = PUNCH_HOLD_FRAMES
                        if game_state_mode == "PLAYING" and p1_hp > 0 and p2_hp > 0:
                            if p1_punch_rect.colliderect(p2_body_rect):
                                if current_state.get("2P") and current_state["2P"]["defend"]:
                                    p2_state["guard_text_timer"] = EFFECT_DURATION
                                    p1_state["stun_timer"] = 30 
                                    p1_state["punch_timer"] = 0 
                                    p1_move, p1_is_defend = "idle", False
                                    p1_combo = 0 
                                else:
                                    p1_combo += 1
                                    p1_combo_timer = 90
                                    if p1_combo == 3: p1_state["energy"] = min(5, p1_state["energy"] + 1)

                                    if p1_state["energy"] >= 5:
                                        p2_hp = max(0, p2_hp - 20)
                                        p1_state["energy"] = 0 
                                        p1_state["crit_text_timer"] = EFFECT_DURATION * 3
                                        p2_state["hit_spark_timer"] = EFFECT_DURATION * 2 
                                        p2_state["pos_x"] += 120 
                                        
                                        right_bound = SCREEN_WIDTH - player_w//2
                                        if p2_state["pos_x"] >= right_bound:
                                            p2_state["pos_x"] = right_bound - 25 
                                            p2_state["stun_timer"] = 20         
                                            p2_wall_bounce_text = 20            
                                        
                                        hitstop_timer = 12
                                        shake_timer = 15
                                        shake_intensity = 10
                                    else:
                                        p2_hp = max(0, p2_hp - 10)
                                        p1_state["energy"] = min(5, p1_state["energy"] + 1)
                                        p2_state["hit_spark_timer"] = EFFECT_DURATION
                                        p2_state["pos_x"] += 30
                                        
                                        hitstop_timer = 4
                                        shake_timer = 6
                                        shake_intensity = 4
                else:
                    p1_move, p1_is_defend = "idle", False

                if current_state.get("2P") and p2_state["stun_timer"] == 0:
                    p2_move = current_state["2P"]["move"]
                    p2_is_defend = current_state["2P"]["defend"]
                    if current_state["2P"]["attack"] == "punch" and p2_state["punch_timer"] == 0:
                        p2_state["punch_timer"] = PUNCH_HOLD_FRAMES
                        if game_state_mode == "PLAYING" and p1_hp > 0 and p2_hp > 0:
                            if p2_punch_rect.colliderect(p1_body_rect):
                                if p1_is_defend:
                                    p1_state["guard_text_timer"] = EFFECT_DURATION
                                    p2_state["stun_timer"] = 30
                                    p2_state["punch_timer"] = 0 
                                    p2_move, p2_is_defend = "idle", False
                                    p2_combo = 0
                                else:
                                    p2_combo += 1
                                    p2_combo_timer = 90
                                    if p2_combo == 3: p2_state["energy"] = min(5, p2_state["energy"] + 1)

                                    if p2_state["energy"] >= 5:
                                        p1_hp = max(0, p1_hp - 20)
                                        p2_state["energy"] = 0
                                        p2_state["crit_text_timer"] = EFFECT_DURATION * 3
                                        p1_state["hit_spark_timer"] = EFFECT_DURATION * 2
                                        p1_state["pos_x"] -= 120 
                                        
                                        left_bound = player_w//2
                                        if p1_state["pos_x"] <= left_bound:
                                            p1_state["pos_x"] = left_bound + 25
                                            p1_state["stun_timer"] = 20
                                            p1_wall_bounce_text = 20
                                        
                                        hitstop_timer = 12
                                        shake_timer = 15
                                        shake_intensity = 10
                                    else:
                                        p1_hp = max(0, p1_hp - 10)
                                        p2_state["energy"] = min(5, p2_state["energy"] + 1)
                                        p1_state["hit_spark_timer"] = EFFECT_DURATION
                                        p1_state["pos_x"] -= 30
                                        
                                        hitstop_timer = 4
                                        shake_timer = 6
                                        shake_intensity = 4
                else:
                    p2_move, p2_is_defend = "idle", False

            if p1_state["stun_timer"] > 0: p1_move, p1_is_defend = "idle", False
            if p2_state["stun_timer"] > 0: p2_move, p2_is_defend = "idle", False

            if p1_hp > 0 and p2_hp > 0:
                if p1_move == "forward": p1_state["pos_x"] += MOVE_SPEED
                elif p1_move == "backward": p1_state["pos_x"] -= MOVE_SPEED
                if p2_move == "forward": p2_state["pos_x"] -= MOVE_SPEED
                elif p2_move == "backward": p2_state["pos_x"] += MOVE_SPEED

            if p2_state["pos_x"] - p1_state["pos_x"] < MIN_DISTANCE:
                mid = (p1_state["pos_x"] + p2_state["pos_x"]) / 2
                p1_state["pos_x"] = mid - (MIN_DISTANCE / 2)
                p2_state["pos_x"] = mid + (MIN_DISTANCE / 2)

            p1_state["pos_x"] = max(player_w//2, min(SCREEN_WIDTH - player_w//2, p1_state["pos_x"]))
            p2_state["pos_x"] = max(player_w//2, min(SCREEN_WIDTH - player_w//2, p2_state["pos_x"]))

            p1_punching = p1_state["punch_timer"] > 0
            if p1_punching: p1_state["punch_timer"] -= 1
            p2_punching = p2_state["punch_timer"] > 0
            if p2_punching: p2_state["punch_timer"] -= 1

        offset_x, offset_y = 0, 0
        if shake_timer > 0:
            shake_timer -= 1
            offset_x = random.randint(-shake_intensity, shake_intensity)
            offset_y = random.randint(-int(shake_intensity*0.6), int(shake_intensity*0.6))

        screen.blit(bg_img, (offset_x, offset_y))

        fps_text = f"FPS: {int(clock.get_fps())} | Inf: {current_inf_time:.1f} ms"
        screen.blit(debug_font.render(fps_text, True, (255, 255, 255)), (10 + offset_x, 2 + offset_y))

        if game_state_mode == "WAITING_START":
            countdown_frames -= 1
            seconds_left = countdown_frames // FPS
            if seconds_left == 3: txt = "3"
            elif seconds_left == 2: txt = "2"
            elif seconds_left == 1: txt = "1"
            else: txt = "F  I  G  H  T  !  !"

            c_surf = countdown_font.render(txt, True, (255, 255, 255))
            screen.blit(c_surf, (SCREEN_WIDTH//2 - c_surf.get_width()//2 + offset_x, SCREEN_HEIGHT//2 - c_surf.get_height()//2 + offset_y))

            if countdown_frames <= 0:
                game_state_mode = "PLAYING"

        elif game_state_mode == "PLAYING":
            bar_w = int(SCREEN_WIDTH * 0.35)
            pygame.draw.rect(screen, (74, 79, 99), (30 + offset_x, 20 + offset_y, bar_w, 20))
            pygame.draw.rect(screen, (74, 79, 99), (SCREEN_WIDTH - 30 - bar_w + offset_x, 20 + offset_y, bar_w, 20))
            pygame.draw.rect(screen, (231, 76, 60), (30 + offset_x, 20 + offset_y, int(bar_w * (p1_hp_buffer / 100)), 20))
            p2_buf_w = int(bar_w * (p2_hp_buffer / 100))
            pygame.draw.rect(screen, (231, 76, 60), (SCREEN_WIDTH - 30 - p2_buf_w + offset_x, 20 + offset_y, p2_buf_w, 20))
            pygame.draw.rect(screen, (46, 204, 113), (30 + offset_x, 20 + offset_y, int(bar_w * (p1_hp / 100)), 20))
            p1_current_w = int(bar_w * (p2_hp / 100))
            pygame.draw.rect(screen, (46, 204, 113), (SCREEN_WIDTH - 30 - p1_current_w + offset_x, 20 + offset_y, p1_current_w, 20))

            pygame.draw.rect(screen, (50, 53, 65), (30 + offset_x, 43 + offset_y, bar_w, 8))
            pygame.draw.rect(screen, (50, 53, 65), (SCREEN_WIDTH - 30 - bar_w + offset_x, 43 + offset_y, bar_w, 8))
            pygame.draw.rect(screen, (241, 196, 15), (30 + offset_x, 43 + offset_y, int(bar_w * (p1_state["energy"] / 5)), 8))
            p2_energy_w = int(bar_w * (p2_state["energy"] / 5))
            pygame.draw.rect(screen, (241, 196, 15), (SCREEN_WIDTH - 30 - p2_energy_w + offset_x, 43 + offset_y, p2_energy_w, 8))

            screen.blit(debug_font.render(f"P1 HP: {int(p1_hp)}", True, (255, 255, 255)), (180 + offset_x, 2 + offset_y))
            screen.blit(debug_font.render(f"P2 HP: {int(p2_hp)}", True, (255, 255, 255)), (SCREEN_WIDTH - 120 + offset_x, 2 + offset_y))

            if p1_combo >= 2:
                screen.blit(title_font.render(f"{p1_combo} HITS!", True, (255, 255, 255)), (30 + offset_x, 55 + offset_y))
            if p2_combo >= 2:
                p2_cb_surf = title_font.render(f"{p2_combo} HITS!", True, (255, 255, 255))
                screen.blit(p2_cb_surf, (SCREEN_WIDTH - 30 - p2_cb_surf.get_width() + offset_x, 55 + offset_y))

            if p1_wall_bounce_text > 0:
                p1_wall_bounce_text -= 1
                screen.blit(title_font.render("WALL CRASH!!", True, (255, 255, 255)), (30 + offset_x, 95 + offset_y))
            if p2_wall_bounce_text > 0:
                p2_wall_bounce_text -= 1
                p2_wc_surf = title_font.render("WALL CRASH!!", True, (255, 255, 255))
                screen.blit(p2_wc_surf, (SCREEN_WIDTH - 30 - p2_wc_surf.get_width() + offset_x, 95 + offset_y))

            p1_draw_x = p1_state["pos_x"] - player_w//2
            if current_state.get("1P"):
                pygame.draw.rect(screen, (46, 204, 113), (p1_draw_x + offset_x, player_y - 25 + offset_y, player_w, 20)) 
                screen.blit(debug_font.render("Player 1", True, (255, 255, 255)), (p1_draw_x + 5 + offset_x, player_y - 23 + offset_y))
            if p1_punching: screen.blit(p1_punch_img, (p1_draw_x + offset_x, player_y + offset_y))
            elif p1_is_defend: screen.blit(p1_defend_img, (p1_draw_x + offset_x, player_y + offset_y))
            else: screen.blit(p1_idle_img, (p1_draw_x + offset_x, player_y + offset_y))

            p2_draw_x = p2_state["pos_x"] - player_w//2
            if current_state.get("2P"):
                pygame.draw.rect(screen, (230, 126, 34), (p2_draw_x + offset_x, player_y - 25 + offset_y, player_w, 20)) 
                screen.blit(debug_font.render("Player 2", True, (255, 255, 255)), (p2_draw_x + 5 + offset_x, player_y - 23 + offset_y))
            if p2_punching: screen.blit(p2_punch_img, (p2_draw_x - 50 + offset_x, player_y + offset_y))
            elif p2_is_defend: screen.blit(p2_defend_img, (p2_draw_x + offset_x, player_y + offset_y))
            else: screen.blit(p2_idle_img, (p2_draw_x + offset_x, player_y + offset_y))

            p2_is_crit = p2_state["crit_text_timer"] > 0
            if p1_state["stun_timer"] > 0:
                screen.blit(title_font.render("STUNNED!", True, (255, 255, 255)), (p1_state["pos_x"] - 65 + offset_x, player_y - 65 + offset_y))
            if p1_state["hit_spark_timer"] > 0:
                p1_state["hit_spark_timer"] -= 1
                rad = 65 if p2_is_crit else 35
                pygame.draw.circle(screen, (255, 69, 0) if p2_is_crit else (241, 196, 15), (int(p1_state["pos_x"]) + offset_x, player_y + 40 + offset_y), rad)
                pygame.draw.circle(screen, (255, 255, 255), (int(p1_state["pos_x"]) + offset_x, player_y + 40 + offset_y), 20)
                
                t_msg = "CRITICAL BREAK!! -20" if p2_is_crit else "HIT!! -10"
                screen.blit(title_font.render(t_msg, True, (255, 255, 255)), (p1_state["pos_x"] - 110 + offset_x, player_y - 70 + offset_y))
            if p1_state["guard_text_timer"] > 0:
                p1_state["guard_text_timer"] -= 1
                screen.blit(title_font.render("GUARD!", True, (255, 255, 255)), (p1_state["pos_x"] - 50 + offset_x, player_y - 70 + offset_y))

            p1_is_crit = p1_state["crit_text_timer"] > 0
            if p2_state["stun_timer"] > 0:
                screen.blit(title_font.render("STUNNED!", True, (255, 255, 255)), (p2_state["pos_x"] - 65 + offset_x, player_y - 65 + offset_y))
            if p2_state["hit_spark_timer"] > 0:
                p2_state["hit_spark_timer"] -= 1
                rad = 65 if p1_is_crit else 35
                pygame.draw.circle(screen, (255, 69, 0) if p1_is_crit else (241, 196, 15), (int(p2_state["pos_x"]) + offset_x, player_y + 40 + offset_y), rad)
                pygame.draw.circle(screen, (255, 255, 255), (int(p2_state["pos_x"]) + offset_x, player_y + 40 + offset_y), 20)
                
                t_msg = "CRITICAL BREAK!! -20" if p1_is_crit else "HIT!! -10"
                screen.blit(title_font.render(t_msg, True, (255, 255, 255)), (p2_state["pos_x"] - 110 + offset_x, player_y - 70 + offset_y))
            if p2_state["guard_text_timer"] > 0:
                p2_state["guard_text_timer"] -= 1
                screen.blit(title_font.render("GUARD!", True, (255, 255, 255)), (p2_state["pos_x"] - 50 + offset_x, player_y - 70 + offset_y))

            if p1_hp <= 0 or p2_hp <= 0:
                if game_over_timer == 0: game_over_timer = FPS * 3 
                w_txt = "PLAYER 1 WINS!" if p2_hp <= 0 else "PLAYER 2 WINS!"
                m_surf = title_font.render(w_txt, True, (255, 255, 255)) 
                screen.blit(m_surf, (SCREEN_WIDTH//2 - m_surf.get_width()//2 + offset_x, SCREEN_HEIGHT//2 - 20 + offset_y))
                game_over_timer -= 1
                
                if game_over_timer <= 0:
                    p1_hp, p2_hp = 100, 100
                    p1_hp_buffer, p2_hp_buffer = 100, 100
                    p1_combo, p2_combo = 0, 0
                    p1_state = {"pos_x": 150, "punch_timer": 0, "hit_spark_timer": 0, "guard_text_timer": 0, "stun_timer": 0, "energy": 0, "crit_text_timer": 0}
                    p2_state = {"pos_x": SCREEN_WIDTH - 150, "punch_timer": 0, "hit_spark_timer": 0, "guard_text_timer": 0, "stun_timer": 0, "energy": 0, "crit_text_timer": 0}
                    countdown_frames = FPS * 4
                    game_state_mode = "WAITING_START"
                    game_over_timer = 0

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
