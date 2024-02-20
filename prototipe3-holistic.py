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

MODELTORUN = 'pose' # OR 'holistic' (holistic includes hands)

VIDEOSOURCE = 2 # you have to work out this number
joint = 0

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

######################################################################
# POSE ONLY
######################################################################
mp_pose = mp.solutions.pose
model_path = "pose_landmarker_full.task"
#model_path = "pose_landmarker_heavy.task"
num_poses = 3
to_window = None
to_window2 = None
last_timestamp_ms = 0


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
      right_wrist = pose_bodies[mp_pose.PoseLandmark.RIGHT_WRIST.value]

      if right_wrist.visibility > 0.8 and cameraID == 0:
        print(f'right wrist: {right_wrist.x:.5f},{right_wrist.y:.5f}, {right_wrist.z:.5f}')

      ######################################################################
      ## ONLY SEND HIPS in 3D SPACE
      ######################################################################
      acceptableVisiblity = 0#.8
      if(left_hip.visibility > acceptableVisiblity and right_hip.visibility > acceptableVisiblity):

        
        ## found the hips so send rest of body
        if results.pose_world_landmarks:
          node = 0
          # TEMP currentbody = results.pose_world_landmarks[bodycount]
          currentbody = results.pose_landmarks[bodycount]

          for lm in currentbody:
            if node == 23:
              msg = str(bodycount) + ',' + str(23) +  f',{left_hip.x:.5f},{left_hip.y:.5f}, {left_hip.z:.5f}, {left_hip.visibility:.3f}'+ ',' + str(cameraID)
              client.sendall(msg.encode("utf-8"))
              #print(msg)
            elif node == 24:
              msg = str(bodycount) + ',' + str(24) +  f',{right_hip.x:.5f},{right_hip.y:.5f}, {right_hip.z:.5f}, {right_hip.visibility:.3f}'+ ',' + str(cameraID)
              client.sendall(msg.encode("utf-8"))
            else:
              msg = str(bodycount) + ',' + str(node) +  f',{lm.x:.5f},{lm.y:.5f}, {lm.z:.5f}, {lm.visibility:.3f}'+ ',' + str(cameraID)
              client.sendall(msg.encode("utf-8"))
            node = node + 1
      bodycount = bodycount + 1

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
if(MODELTORUN == 'pose'):
  print('go')
  runBothCams = True
  with vision.PoseLandmarker.create_from_options(options) as landmarker:
    with vision.PoseLandmarker.create_from_options(options) as landmarker2:
      cap = cv2.VideoCapture(1)
      cap2 = None
      if(runBothCams):
        cap2 = cv2.VideoCapture(2)
      
      # Create a loop to read the latest frame from the camera using VideoCapture read()
      while cap.isOpened() and cap2.isOpened():
      #while cap.isOpened():
          success, image = cap.read()
          if not success:
              print("Image capture 1 failed.")
              break
          if(runBothCams):
            success, image2 = cap2.read()
            if not success:
                print("Image capture 2 failed.")
                break

          # Convert the frame received from OpenCV to a MediaPipe’s Image object.
          mp_image = mp.Image(
              image_format=mp.ImageFormat.SRGB,
              data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

          # keep for async tests later, might be faster
          # landmarker.detect_async(mp_image, timestamp_ms)

          results = landmarker.detect_for_video(mp_image, timestamp_ms)
          find_and_package_bodies(results, 0)

          if(runBothCams):
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image2 = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            #timestamp_ms2 = timestamp_ms + 1

            results2 = landmarker2.detect_for_video(mp_image2, timestamp_ms)
            find_and_package_bodies(results2, 1)

          to_window = cv2.cvtColor(
            draw_landmarks_on_image(mp_image.numpy_view(), results), cv2.COLOR_RGB2BGR)
          if to_window is not None:
            cv2.imshow('PoseFinder', to_window)

          if(runBothCams):
            to_window2 = cv2.cvtColor(
              draw_landmarks_on_image(mp_image2.numpy_view(), results2), cv2.COLOR_RGB2BGR)
            if to_window2 is not None:
              cv2.imshow('PoseFinder2', to_window2)

          if cv2.waitKey(5) & 0xFF == 27:
            break


######################################################################
# Holistic ONLY
######################################################################
mp_holistic = mp.solutions.holistic

# From https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md

if MODELTORUN == 'holistic':
  cap = cv2.VideoCapture(0)
  # For HOLISTIC:
  with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      model_complexity=1) as holistic:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)
      
      if results.pose_world_landmarks:
        joint = 0
        for lm in results.pose_world_landmarks.landmark:
          msg = str(joint) + ',' + str(f'{lm.x:.5f}') + ',' + str(f'{lm.y:.5f}') + ',' + str(f'{lm.z:.5f}') + ',' + str(f'{lm.visibility:.3f}')# + 'end'
          #msg = str(node) + '\n' + str(lm)
          print(msg)
          client.sendall(msg.encode("utf-8"))
          joint = joint + 1
      
      #client.send('\nend\n'.encode("utf-8")[:1024])
      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

# Exit program gracefully
cap.release()
client.close()