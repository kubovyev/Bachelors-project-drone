#!/usr/bin/python3

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pickle
from torchvision import transforms
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import torchvision.transforms as transforms
import numpy as np
from std_msgs.msg import Float32
from mrs_msgs.msg import UavState
from mrs_msgs.msg import VelocityReferenceStamped
from mrs_msgs.srv import ReferenceStampedSrv
from mrs_msgs.srv import ReferenceStampedSrvRequest
from mrs_msgs.msg import PositionCommand
from mrs_msgs.msg import Float64Stamped
import random
from torchvision.utils import save_image
import message_filters


''' Set seed to fix number for repeatability '''
torch.manual_seed(1)
uav = '/uav11/'

# Instantiate CvBridge
bridge = CvBridge()

# pub = rospy.Publisher(uav + 'control_manager/velocity_reference', VelocityReferenceStamped, queue_size=10)
raw_prediction_left = rospy.Publisher(uav + 'neural_network/raw_left', Float32, queue_size=10)
raw_prediction_straight = rospy.Publisher(uav + 'neural_network/raw_straight', Float32, queue_size=10) 
raw_prediction_right = rospy.Publisher(uav + 'neural_network/raw_right', Float32, queue_size=10)
filtered_prediction_left = rospy.Publisher(uav + 'neural_network/filter_left', Float32, queue_size=10)
filtered_prediction_straight = rospy.Publisher(uav + 'neural_network/filter_straight', Float32, queue_size=10) 
filtered_prediction_right = rospy.Publisher(uav + 'neural_network/filter_right', Float32, queue_size=10) 



filter_len = 15
left_arr = [0 for i in range(filter_len)]
straight_arr = [0 for i in range(filter_len)]
right_arr = [0 for i in range(filter_len)]





def compute(img, hdg):
    # Converting ROS Image message to OpenCV2, shaping and then converting to tensor
    cv2_img = img
    cv2_img = cv2.flip(cv2_img, -1)  # Because the camera is upside down!
    cv2_img = cv2.resize(cv2_img, (101, 101), interpolation=cv2.INTER_AREA)
    cv2_img = np.asarray(cv2_img)
    tensor = torch.tensor(cv2_img, dtype=torch.float).permute(2, 0, 1)
    tensor = torch.unsqueeze(tensor, dim=0)
    # Feed the tensor to the neural network
    prediction = model(tensor)
    prediction = prediction.detach().cpu().numpy()
    prediction = prediction[0]
    left = prediction[0]
    straight = prediction[1]
    right = prediction[2]
    # Arrays needed for the moving average
    left_arr.append(left)
    left_arr.pop(0)
    straight_arr.append(straight)
    straight_arr.pop(0)
    right_arr.append(right)
    right_arr.pop(0)
    # Moving average calculation
    left = sum(left_arr)/len(left_arr)
    straight = sum(straight_arr)/len(straight_arr)
    right = sum(right_arr)/len(right_arr)
    # Limiting the predictions just to be sure that they do not exceed the range [0, 1]
    if left > 1:
        left = 1
    elif left < 0:
        left = 0
    if straight > 1:
        straight = 1
    elif straight < 0:
        straight = 0
    if right > 1:
        right = 1
    elif right < 0:
        right = 0

    # print(f'{left:.4f}',f'{straight:.4f}', f'{right:.4f}', end = '\r')
    rospy.loginfo('{:.4f}, {:.4f}, {:.4f}'.format(left, straight, right))

    speed_straight = 1  # Maximum forward speed
    speed_angular = 0.2  # Maximum angular speed
    rate_weight = 1
    message = VelocityReferenceStamped()
    #  Creating messages for the raw data
    raw_left_msg = Float32()
    raw_left_msg = prediction[0]
    raw_straight_msg = Float32()
    raw_straight_msg = prediction[1]
    raw_right_msg = Float32()
    raw_right_msg = prediction[2]
    #  Publishing raw predictions
    raw_prediction_left.publish(raw_left_msg)
    raw_prediction_straight.publish(raw_straight_msg)
    raw_prediction_right.publish(raw_right_msg)
    #  Creating messages for the filtered data
    left_msg = Float32()
    left_msg = left
    straight_msg = Float32()
    straight_msg = straight
    right_msg = Float32()
    right_msg = right
    #  Publishing filtered
    filtered_prediction_left.publish(left_msg)
    filtered_prediction_straight.publish(straight_msg)
    filtered_prediction_right.publish(right_msg)

    message.reference.use_heading_rate = 1
    message.header.frame_id = uav[1:] + "fcu_untilted"
    rate = (right-left) * speed_angular
    # PART FOR VELOCITY GENERATION
    # message.reference.heading_rate = rate
    # message.reference.velocity.x = speed_straight*straight
    # pub.publish(message)

    #  Trajectory message generation and publishing
    trajectory_msg = ReferenceStampedSrvRequest()
    trajectory_msg.header.frame_id = uav[1:] + "fcu_untilted"

    length = 3
    min_length = 0.45
    if straight > 0.5:  #  Threshold
        trajectory_msg.reference.position.x = length * math.cos(rate) * straight
        trajectory_msg.reference.position.y = length * math.sin(rate) * straight
    else:
        trajectory_msg.reference.position.x = min_length * math.cos(rate)
        trajectory_msg.reference.position.y = min_length * math.sin(rate)

    global height

    desired_z = 1.8 - height

    # rospy.loginfo('desired_z: {}'.format(desired_z))

    trajectory_msg.reference.position.z = desired_z
    trajectory_msg.reference.heading = rate * rate_weight
    trajectory_client(trajectory_msg)


heading = None
height = None

def image_callback(img_msg):
    image = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    compute(image, heading)

def height_callback(height_msg):
    global height
    height = height_msg.value

def heading_callback(hdg_msg):
    global heading
    heading = hdg_msg
    rospy.loginfo_once('getting height')

def trajectory_client(reference):
    rospy.wait_for_service(uav + 'pathfinder/reference')
    try:
        function = rospy.ServiceProxy(uav + 'pathfinder/reference', ReferenceStampedSrv)
        resp1 = function(reference)
        if not resp1.success:
          rospy.logerr_throttle(1.0, "pathfinder failed: {}".format(resp1.message))
        else:
          rospy.loginfo_once("succeeded setting path")
        return None
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


''' Neural network '''

class Conv_Model(torch.nn.Module):
    def __init__(self, nbr_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(4, 4), padding=0)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 4), padding=0)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 4), padding=0)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(4, 4), padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.lin = nn.Linear(512, 200)
        self.lin2 = nn.Linear(200, nbr_classes)
        self.weight_init()

    def forward(self, x):
        x = x/255*2-1
        x = self.conv1(x)
        x = self.maxp1(x)
        x = nn.functional.tanh(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = nn.functional.tanh(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = nn.functional.tanh(x)
        x = self.conv4(x)
        x = self.maxp4(x)
        x = nn.functional.tanh(x)
        x = self.lin(x.view(-1, self.lin.in_features))
        x = self.lin2(x.view(-1, self.lin2.in_features))
        x = self.softmax(x)
        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)


# Neural network is 3-class
nbr_cls = 3
CLS = {
    0: 'left',
    1: 'straight',
    2: 'right'
}


def main():
    rospy.init_node('listener')
    print("Node initialized")

    # Define your image topic
    image_topic = uav + "rgbd/color/image_raw"
    heading_topic = uav + "control_manager/position_cmd"

    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(uav + "odometry/height", Float64Stamped, height_callback)
    print("Subscibers initialized")
    rospy.spin()

if __name__ == '__main__':
    print("Path follower V1.0")
    #Modules
    model = Conv_Model(nbr_classes=nbr_cls)
    model.load_state_dict(torch.load(os.path.abspath(os.path.dirname(__file__)) + '/weights_nl_final.pth', map_location='cpu'))
    np.set_printoptions(suppress=True)
    main()
