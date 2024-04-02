import cv2
import mediapipe as mp
import numpy as np
import socket
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import protobuf_to_dict
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

######################################################################
# SOCKET SETUP
######################################################################

try:
  client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  server_ip = "127.0.0.1"  # replace with the server's IP address
  server_port = 8888  # replace with the server's port number
  client.connect((server_ip, server_port))
  print('Socket Setup Success')
except socket.error as err:
  print(err)

mp_pose = mp.solutions.pose
model_path = "pose_landmarker_full.task"
#model_path = "pose_landmarker_heavy.task" #heavy is slower and more accurate
num_poses = 2 # maximum number of poses that can be tracked

to_window = None
to_window2 = None
last_timestamp_ms = 0

# Visualiser function
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def find_and_package_bodies(results, cameraID):
  if results.pose_landmarks:
    #get all found poses
    bodycount = 0

    for pose_bodies in results.pose_landmarks:
      # hips are 23 and 23
      left_hip = pose_bodies[mp_pose.PoseLandmark.LEFT_HIP.value]
      right_hip = pose_bodies[mp_pose.PoseLandmark.RIGHT_HIP.value]
      # Pose landmark x,y gives the coordinate on the screen
      right_wrist = pose_bodies[mp_pose.PoseLandmark.RIGHT_WRIST.value]
      left_wrist = pose_bodies[mp_pose.PoseLandmark.LEFT_WRIST.value]

      # Poseworldlandmark gives coordinates relative to hips
      # right_wrist_world = results.pose_world_landmarks[bodycount][mp_pose.PoseLandmark.RIGHT_WRIST.value]

      # For testing / debugging only send wrists so is more obvious what is happening
      if right_wrist.visibility > 0.8:
        print(f'Cam0 right wrist: {right_wrist.x:.5f},{right_wrist.y:.5f}, {right_wrist.z:.5f}')
        #print(f'Cam0 right wrist: {right_wrist.x:.5f},{right_wrist_world.x:.5f}')
        msg = str(bodycount) + ',' + str(16) +  f',{right_wrist.x:.5f},{right_wrist.y:.5f}, {right_wrist.z:.5f}, {right_wrist.visibility:.3f}'+ ',' + str(cameraID)
        #client.sendall(msg.encode("utf-8"))

      if left_wrist.visibility > 0.8:
        print(f'Cam1 left wrist: {left_wrist.x:.5f},{left_wrist.y:.5f}, {left_wrist.z:.5f}')
        msg = str(bodycount) + ',' + str(17) +  f',{left_wrist.x:.5f},{left_wrist.y:.5f}, {left_wrist.z:.5f}, {left_wrist.visibility:.3f}'+ ',' + str(cameraID)
        #client.sendall(msg.encode("utf-8"))

      ######################################################################
      ## Main joint packaging section (to send everything later)
      ######################################################################
      acceptableVisiblity = 0.2
      if(left_hip.visibility > acceptableVisiblity and right_hip.visibility > acceptableVisiblity):
        ## found the hips so send rest of body
        if results.pose_world_landmarks:
          node = 0
          
          # currentbody = results.pose_world_landmarks[bodycount]
          currentbody = results.pose_landmarks[bodycount]

          for lm in currentbody:
            msg = str(bodycount) + ',' + str(node) +  f',{lm.x:.5f},{lm.y:.5f}, {lm.z:.5f}, {lm.visibility:.3f}'+ ',' + str(cameraID)
            client.sendall(msg.encode("utf-8"))
            # This code checks for hips specifically
            # if node == 23:
            #   msg = str(bodycount) + ',' + str(23) +  f',{left_hip.x:.5f},{left_hip.y:.5f}, {left_hip.z:.5f}, {left_hip.visibility:.3f}'+ ',' + str(cameraID)
            #   #client.sendall(msg.encode("utf-8"))
            #   #print(msg)
            # elif node == 24:
            #   msg = str(bodycount) + ',' + str(24) +  f',{right_hip.x:.5f},{right_hip.y:.5f}, {right_hip.z:.5f}, {right_hip.visibility:.3f}'+ ',' + str(cameraID)
            #   #client.sendall(msg.encode("utf-8"))
            # else:
            #   msg = str(bodycount) + ',' + str(node) +  f',{lm.x:.5f},{lm.y:.5f}, {lm.z:.5f}, {lm.visibility:.3f}'+ ',' + str(cameraID)
            #   client.sendall(msg.encode("utf-8"))
            node = node + 1
      bodycount = bodycount + 1

# Setup configuration options for mediapipe pose tracking
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=num_poses,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

# For POSE ONLY:
# Multi camera stuff is hardcoded for now, but functions need to be extracted later
runBothCams = True

with vision.PoseLandmarker.create_from_options(options) as landmarker:
  with vision.PoseLandmarker.create_from_options(options) as landmarker2:
    cap = cv2.VideoCapture(0)
    print()
    cap2 = None
    if(runBothCams): 
      cap2 = cv2.VideoCapture(1)
    # Create a loop to read the latest frame from the camera using VideoCapture read()
    while cap.isOpened() and cap2.isOpened():
    #while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture 1 failed.")
            continue
        if(runBothCams):
          success, image2 = cap2.read()
          if not success:
              print("Image capture 2 failed.")
              continue

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Runs the pose dections on First Camera
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        find_and_package_bodies(results, 0)

        if(runBothCams):
          # Convert the frame received from OpenCV to a MediaPipe’s Image object.
          mp_image2 = mp.Image(
              image_format=mp.ImageFormat.SRGB,
              data=cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
          #timestamp_ms2 = timestamp_ms + 1

          # Runs the pose dections on Second Camera
          results2 = landmarker2.detect_for_video(mp_image2, timestamp_ms)
          find_and_package_bodies(results2, 1)

        to_window = cv2.cvtColor(
          draw_landmarks_on_image(mp_image.numpy_view(), results), cv2.COLOR_RGB2BGR)
        if to_window is not None:
          cv2.imshow('PoseFinder 0', to_window)

        if(runBothCams):
          to_window2 = cv2.cvtColor(
            draw_landmarks_on_image(mp_image2.numpy_view(), results2), cv2.COLOR_RGB2BGR)
          if to_window2 is not None:
            cv2.imshow('PoseFinder 1', to_window2)

        if cv2.waitKey(5) & 0xFF == 27:
          break

# Exit program gracefully
cap.release()
if(runBothCams): 
  cap2.release()
client.close()