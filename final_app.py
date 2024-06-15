## om namah shivay

## imports
from vtk import vtkObject ,vtkWindowToImageFilter,vtkGLTFImporter,vtkRenderer,vtkRenderWindow,vtkTransform
from PIL import Image
import streamlit as st
import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,VideoTransformerBase, WebRtcMode
import streamlit_webrtc
import av




from twilio.rest import Client


@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
     
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers


## class for transforming and showing image
class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):

        # def logic(image):
        #     global img_to_stack,counter,output_img
        #     img_h,img_w,img_c=image.shape
        #     results=face_mesh.process(image)
        #     face_2d=[]
        #     if results.multi_face_landmarks:
        #         counter+=1
        #         counter%=1
        #         for face_landmarks in results.multi_face_landmarks:
        #             idx=0
        #             ## indexes of 6 points used during training
        #             for idx in [1,199,263,33,291,61]:
        #                 lm= face_landmarks.landmark[idx]
        #                 x,y=int(lm.x*img_w),int(lm.y*img_h)
        #                 face_2d.append([x,y])
        #                 cv2.circle(image,(x,y),2,(0,255,0),thickness=-1)
        #             if counter==0:
        #                 face_2d=np.array(face_2d,dtype=np.float64)
        #                 data=face_2d.flatten()
        #                 data=np.reshape(data,(1,12,))
        #                 label=headpose_model.predict(data,verbose=0)[0]
        #                 deg_x,deg_y,deg_z,t_x,t_y,t_z=label
        #                 img_to_stack=rotate(deg_x,deg_y,deg_z) 
        #                 output_img=np.vstack((image,img_to_stack)) 
        #                 output_img=cv2.resize(output_img,(640,480))
        #                 output_img=output_img.astype("uint8")
        #     return output_img
        # try:
        #     frm=frame.to_ndarray(format="bgr24")
        #     image=cv2.flip(frm,1)
        #     image2=logic(image)
        #     return av.VideoFrame.from_ndarray(image2,format="bgr24")
        # except:
        #     return av.VideoFrame.from_ndarray(output_img,format="bgr24")
        return av.VideoFrame.from_ndarray(frame,format="bgr24")



def main():
    st.title("Real-time Video Stream using WebRTC")
    vtkObject.GlobalWarningDisplayOff() 
    # webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


## function for rotating iron man mask model through specified angles
# the overall idea is to get back model to its original position by using identity
# then give absolute rotations in all three directions 

def rotate(angle_x,angle_y,angle_z):
    global pre_x,pre_y,pre_z,transform,imported_actors,actor,render_window,image_holder_frame
    transform.Identity()
    transform.RotateX(-angle_x-pre_x)
    transform.RotateY(-angle_z-pre_z)
    transform.RotateZ(-angle_y-pre_y)
    imported_actors.InitTraversal()
    actor = imported_actors.GetNextActor()
    while actor:
        actor.SetUserTransform(transform)
        actor = imported_actors.GetNextActor()
    render_window.Render()
    window_to_image_filter = vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    # Convert vtkImageData to numpy array
    vtk_image = window_to_image_filter.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    vtk_array.SetNumberOfComponents(3)  # Ensure RGB
    np_image = np.array(vtk_array).reshape(height, width, 3)

    # # Convert RGB to BGR
    np_image = np_image[:, :, ::-1]
    return np_image


if __name__=="__main__":
    main_dir=os.path.dirname(__file__)
    image_to_stack=np.zeros((640,480,3))
    output_img=np.zeros((480,640,3),dtype="uint8")
    counter=0
    pre_x,pre_y,pre_z=90,180,0
    image_holder_frame=st.empty()
    # # Load GLB file
    importer = vtkGLTFImporter()
    importer.SetFileName(os.path.join(main_dir,"mark_85.glb"))
    importer.Read()

    # # Create a renderer
    renderer = vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Add the imported actors to the renderer
    imported_actors = importer.GetRenderer().GetActors()
    imported_actors.InitTraversal()
    actor = imported_actors.GetNextActor()
    while actor:
        renderer.AddActor(actor)
        actor = imported_actors.GetNextActor()

    # Create a render window
    render_window = vtkRenderWindow()
    render_window.SetSize(640, 480)
    render_window.OffScreenRenderingOn()
    render_window.AddRenderer(renderer)
    transform = vtkTransform()
    
    headpose_model_json_path=os.path.join(main_dir,"models","head_pose_model.json")
    headpose_model_weights_path=os.path.join(main_dir,"models","head_pose_model_weights.h5")

    ##loading models
    with open(headpose_model_json_path,"r") as file:
      headpose_model=file.read()
      headpose_model=model_from_json(headpose_model)
      headpose_model.load_weights(headpose_model_weights_path)

    mp_face_mesh=mp.solutions.face_mesh
    face_mesh=mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    mp_drawing=mp.solutions.drawing_utils
    drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

    main()