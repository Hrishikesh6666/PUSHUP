# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, FileResponse
# from cvzone.PoseModule import PoseDetector
# from starlette.websockets import WebSocketState
# import cv2
# import numpy as np
# import time
# import uvicorn
# import base64
# import logging
# import json
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# app = FastAPI()
#
# # Mount static files
# # app.mount("/static", StaticFiles(directory="static"), name="static")
#
# @app.get("/")
# async def read_root():
#     try:
#         return FileResponse("static/Home.html")
#     except Exception as e:
#         logger.error(f"Root endpoint error: {str(e)}")
#         return HTMLResponse(f"<h1>Error: {str(e)}</h1>", status_code=500)
#
# class PushupTracker:
#     def __init__(self):
#         self.detector = PoseDetector()
#         self.pushup_count = 0
#         self.set_count = 0
#         self.pushups_in_current_set = 0
#         self.pushup_position = "up"
#         self.threshold_angle = 90
#         self.min_time_between_pushups = 1.5
#         self.last_pushup_time = 0
#         self.feedback = ""
#         # Calibration timer variables
#         self.start_time = time.time()
#         self.calibration_done = False
#
#     def calculate_angle(self, a, b, c):
#         try:
#             angle = np.arctan2(c[2] - b[2], c[1] - b[1]) - np.arctan2(a[2] - b[2], a[1] - b[1])
#             angle = np.abs(angle * 180.0 / np.pi)
#             return angle if angle <= 180 else 360 - angle
#         except Exception as e:
#             logger.error(f"Angle error: {str(e)}")
#             return 90
#
#     async def process_frame(self, img):
#         try:
#             img = self.detector.findPose(img, draw=False)
#             lmList, _ = self.detector.findPosition(img, bboxWithHands=False)
#             self.feedback = ""
#
#             result = {
#                 "pushup_count": self.pushup_count,
#                 "set_count": self.set_count,
#                 "feedback": self.feedback,
#                 "landmarks": [],
#                 "calibration_remaining": 0
#             }
#
#             current_time = time.time()
#
#             # Handle calibration phase
#             if not self.calibration_done:
#                 elapsed = current_time - self.start_time
#                 if elapsed < 5:
#                     remaining = 5 - elapsed
#                     result["calibration_remaining"] = int(remaining)
#                     result["feedback"] = f"Adjust position: {int(remaining)}s"
#                     # Still process landmarks for visualization
#                     if lmList:
#                         result["landmarks"] = [[lm[1], lm[2]] for lm in
#                             [lmList[11], lmList[12], lmList[13], lmList[14], lmList[23], lmList[24]]]
#                     return result
#                 else:
#                     self.calibration_done = True
#                     result["calibration_remaining"] = 0
#                     result["feedback"] = "Start doing push-ups!"
#
#             # Normal processing after calibration
#             if lmList:
#                 try:
#                     shoulder_left = lmList[11]
#                     shoulder_right = lmList[12]
#                     elbow_left = lmList[13]
#                     elbow_right = lmList[14]
#                     hip_left = lmList[23]
#                     hip_right = lmList[24]
#
#                     left_angle = self.calculate_angle(shoulder_left, elbow_left, hip_left)
#                     right_angle = self.calculate_angle(shoulder_right, elbow_right, hip_right)
#
#                     if left_angle < self.threshold_angle and right_angle < self.threshold_angle:
#                         if self.pushup_position == "up" and (current_time - self.last_pushup_time) > self.min_time_between_pushups:
#                             self.pushup_count += 1
#                             self.pushups_in_current_set += 1
#                             self.pushup_position = "down"
#                             self.last_pushup_time = current_time
#                             self.feedback = "Good pushup!"
#                             if self.pushups_in_current_set >= 12:
#                                 self.set_count += 1
#                                 self.pushups_in_current_set = 0
#                                 self.feedback = "Set complete! Rest now."
#                     elif left_angle > self.threshold_angle and right_angle > self.threshold_angle:
#                         self.pushup_position = "up"
#
#                     result["landmarks"] = [[lm[1], lm[2]] for lm in
#                         [shoulder_left, shoulder_right, elbow_left, elbow_right, hip_left, hip_right]]
#                     result["feedback"] = self.feedback
#
#                 except IndexError:
#                     logger.warning("Missing pose landmarks")
#
#             return result
#
#         except Exception as e:
#             logger.error(f"Processing error: {str(e)}")
#             return result
#
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     tracker = PushupTracker()
#     last_processed = 0
#
#     try:
#         while True:
#             try:
#                 data = await websocket.receive_text()
#
#                 if not data.startswith("data:image/"):
#                     continue
#
#                 header, encoded = data.split(",", 1)
#                 img_bytes = base64.b64decode(encoded)
#                 img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#
#                 if img is None or time.time() - last_processed < 0.1:
#                     continue
#
#                 result = await tracker.process_frame(img)
#                 await websocket.send_text(json.dumps(result))
#                 last_processed = time.time()
#
#             except WebSocketDisconnect:
#                 logger.info("Client disconnected")
#                 break
#             except Exception as e:
#                 logger.error(f"WebSocket error: {str(e)}")
#                 if websocket.client_state == WebSocketState.CONNECTED:
#                     await websocket.send_text(json.dumps({"error": "Processing failed"}))
#
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#     finally:
#         if websocket.client_state == WebSocketState.CONNECTED:
#             await websocket.close()
#             logger.info("Connection closed cleanly")
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PushupTracker:
    def __init__(self):
        self.detector = PoseDetector()
        self.pushup_count = 0
        self.set_count = 0
        self.pushups_in_current_set = 0
        self.pushup_position = "up"
        self.threshold_angle = 90
        self.min_time_between_pushups = 1.5
        self.last_pushup_time = 0
        self.feedback = ""
        self.start_time = time.time()
        self.calibration_done = False

    def calculate_angle(self, a, b, c):
        try:
            vec_ab = [b[1] - a[1], b[2] - a[2]]
            vec_cb = [b[1] - c[1], b[2] - c[2]]

            # Handle zero vectors
            mag_ab = np.linalg.norm(vec_ab)
            mag_cb = np.linalg.norm(vec_cb)
            if mag_ab == 0 or mag_cb == 0:
                return 90

            dot = np.dot(vec_ab, vec_cb)
            cosine_angle = dot / (mag_ab * mag_cb)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cosine_angle))
        except Exception as e:
            logger.error(f"Angle calculation error: {str(e)}")
            return 90

    async def process_frame(self, img):
        try:
            if img is None or img.size == 0:
                logger.warning("Received empty image frame")
                return self._error_response("Empty image frame")

            # Convert to RGB for better detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = self.detector.findPose(img_rgb, draw=False)
            lmList, _ = self.detector.findPosition(img_rgb, bboxWithHands=False)

            result = {
                "pushup_count": self.pushup_count,
                "set_count": self.set_count,
                "feedback": self.feedback,
                "landmarks": [],
                "calibration_remaining": 0,
                "error": None
            }

            current_time = time.time()

            if not self.calibration_done:
                elapsed = current_time - self.start_time
                if elapsed < 5:
                    result["calibration_remaining"] = int(5 - elapsed)
                    result["feedback"] = f"Calibrating: {5 - int(elapsed)}s"
                    if lmList:
                        result["landmarks"] = self._get_landmarks(lmList)
                    return result
                self.calibration_done = True
                result["feedback"] = "Start pushups!"
                return result

            if lmList:
                try:
                    left_angle, right_angle = self._calculate_arm_angles(lmList)
                    self._update_pushup_count(current_time, left_angle, right_angle)
                    result["landmarks"] = self._get_landmarks(lmList)
                    result["feedback"] = self.feedback

                except IndexError as e:
                    logger.warning(f"Missing pose landmarks: {str(e)}")
                    result["feedback"] = "Stay in frame!"
                except Exception as e:
                    logger.error(f"Pose processing error: {str(e)}")
                    result["error"] = str(e)

            return result

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return self._error_response(str(e))

    def _error_response(self, message):
        return {
            "pushup_count": self.pushup_count,
            "set_count": self.set_count,
            "feedback": "System error - resetting...",
            "landmarks": [],
            "calibration_remaining": 0,
            "error": message
        }

    # Rest of the class methods remain the same...


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = PushupTracker()
    logger.info("New WebSocket connection established")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.debug("Received WebSocket message")

                if not data.startswith("data:image/"):
                    logger.warning("Received non-image data")
                    continue

                header, encoded = data.split(",", 1)
                img_bytes = base64.b64decode(encoded)

                # Validate image size
                if len(img_bytes) < 1024:
                    logger.warning("Received invalid image data")
                    await websocket.send_json({"error": "Invalid image data"})
                    continue

                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    logger.warning("Failed to decode image")
                    await websocket.send_json({"error": "Image decoding failed"})
                    continue

                result = await tracker.process_frame(img)
                await websocket.send_json(result)
                logger.debug("Sent processing results")

            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    "error": "Processing error",
                    "message": str(e)
                })
                break

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")


# Rest of the file remains the same...


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)