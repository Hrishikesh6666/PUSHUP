from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np
import time
import uvicorn
import base64
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Pushup Counter API is running Main!"}

class PushupTracker:
    def __init__(self):
        self.detector = PoseDetector(detectionCon=0.7, trackCon=0.7)
        self.pushup_count = 0
        self.set_count = 0
        self.pushups_in_current_set = 0
        self.pushup_position = "up"
        self.down_angle = 90
        self.up_angle = 160
        self.min_valid_angle = self.down_angle + 0.7 * (self.up_angle - self.down_angle)
        self.min_time_between_pushups = 1.0
        self.last_pushup_time = 0
        self.feedback = ""
        self.required_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points a, b, c (each as (x,y) tuples) with b as the vertex."""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except Exception as e:
            logger.error(f"Angle calculation error: {str(e)}")
            return 90

    def check_visibility(self, landmarks_dict):
        """Check if all required landmarks are present and visible."""
        for idx in self.required_landmarks:
            if idx not in landmarks_dict:
                return False
            x, y = landmarks_dict[idx][1], landmarks_dict[idx][2]
            if x == 0 or y == 0:
                return False
        return True

    def check_alignment(self, landmarks_dict):
        """Check if shoulders, hips, and ankles are aligned."""
        try:
            mid_shoulder_x = (landmarks_dict[11][1] + landmarks_dict[12][1]) / 2
            mid_shoulder_y = (landmarks_dict[11][2] + landmarks_dict[12][2]) / 2
            mid_hip_x = (landmarks_dict[23][1] + landmarks_dict[24][1]) / 2
            mid_hip_y = (landmarks_dict[23][2] + landmarks_dict[24][2]) / 2
            mid_ankle_x = (landmarks_dict[27][1] + landmarks_dict[28][1]) / 2
            mid_ankle_y = (landmarks_dict[27][2] + landmarks_dict[28][2]) / 2

            mid_shoulder = (mid_shoulder_x, mid_shoulder_y)
            mid_hip = (mid_hip_x, mid_hip_y)
            mid_ankle = (mid_ankle_x, mid_ankle_y)

            angle = self.calculate_angle(mid_shoulder, mid_hip, mid_ankle)
            if angle < 150:
                self.feedback = "Keep body straight - align hips, shoulders, and ankles"
                return False
            return True
        except KeyError as e:
            logger.error(f"Missing landmark for alignment: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Alignment error: {str(e)}")
            return False

    async def process_frame(self, img):
        try:
            img = self.detector.findPose(img, draw=False)
            landmarks, _ = self.detector.findPosition(img, bboxWithHands=True)
            landmarks_dict = {lm[0]: lm for lm in landmarks}

            result = {
                "count": self.pushup_count,
                "sets": self.set_count,
                "feedback": self.feedback,
                "timestamp": time.time_ns()
            }

            if not self.check_visibility(landmarks_dict):
                self.feedback = "Position your full body in frame"
                return result

            if not self.check_alignment(landmarks_dict):
                return result

            # Get elbow angles
            left_shoulder = (landmarks_dict[11][1], landmarks_dict[11][2])
            left_elbow = (landmarks_dict[13][1], landmarks_dict[13][2])
            left_wrist = (landmarks_dict[15][1], landmarks_dict[15][2])
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

            right_shoulder = (landmarks_dict[12][1], landmarks_dict[12][2])
            right_elbow = (landmarks_dict[14][1], landmarks_dict[14][2])
            right_wrist = (landmarks_dict[16][1], landmarks_dict[16][2])
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            avg_angle = (left_angle + right_angle) / 2
            current_time = time.time()

            if self.pushup_position == "up" and avg_angle < self.down_angle:
                self.pushup_position = "down"
                self.feedback = "Lower chest to floor"
            elif self.pushup_position == "down" and avg_angle > self.min_valid_angle:
                if (current_time - self.last_pushup_time) > self.min_time_between_pushups:
                    self.pushup_count += 1
                    self.pushups_in_current_set += 1
                    self.pushup_position = "up"
                    self.last_pushup_time = current_time
                    if avg_angle >= self.up_angle:
                        self.feedback = "Excellent rep! Full extension"
                    else:
                        self.feedback = "Good rep! (≥70% range)"
                    if self.pushups_in_current_set >= 12:
                        self.set_count += 1
                        self.pushups_in_current_set = 0
                        self.feedback = "Set complete! Rest now"
                else:
                    self.feedback = "Don't rush - maintain proper form"
            else:
                if self.pushup_position == "down":
                    self.feedback = "Keep lowering until your chest nearly touches the floor"
                elif self.pushup_position == "up" and avg_angle < self.up_angle:
                    self.feedback = "Fully extend your arms"
                else:
                    self.feedback = "Hold steady"

            result.update({
                "count": self.pushup_count,
                "sets": self.set_count,
                "feedback": self.feedback
            })
            return result

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "count": self.pushup_count,
                "sets": self.set_count,
                "feedback": "System error",
                "timestamp": time.time_ns()
            }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = PushupTracker()
    last_process = 0
    PROCESS_INTERVAL = 0.1  # ~10 FPS

    try:
        while True:
            data = await websocket.receive_text()

            if not data.startswith("data:image/"):
                continue

            current_time = time.time()
            if current_time - last_process < PROCESS_INTERVAL:
                continue

            try:
                _, encoded = data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    result = await tracker.process_frame(img)
                    await websocket.send_json(result)
                    last_process = current_time

            except Exception as e:
                logger.error(f"Frame error: {str(e)}")
                await websocket.send_json({
                    "error": "Processing failed",
                    "count": tracker.pushup_count,
                    "sets": tracker.set_count,
                    "timestamp": time.time_ns()
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"WebSocket already closed: {str(e)}")

if __name__ == "__main__":
    # Get the port from the environment variable, default to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


