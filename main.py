#!/usr/bin/python3

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from mrs_msgs.msg import PositionCommand
import random
from torchvision.utils import save_image
import message_filters


''' Set seed to fix number for repeatability '''
torch.manual_seed(1)
uav = '/uav70/'

# Instantiate CvBridge
bridge = CvBridge()

pub = rospy.Publisher(uav + 'control_manager/velocity_reference', VelocityReferenceStamped, queue_size=10)
filter_len = 15
left_arr = [0 for i in range(filter_len)]
straight_arr = [0 for i in range(filter_len)]
right_arr = [0 for i in range(filter_len)]





def compute(img, hdg):
    ##if hdg is not None:
        ##print(hdg.value)
    ##print("computing")

    ##print("Received an image!")
    # Convert your ROS Image message to OpenCV2
    cv2_img = img
    # cv2_img = cv2.imread("test.jpg")
    # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.resize(cv2_img, (101, 101), interpolation=cv2.INTER_AREA)
    cv2_img = np.asarray(cv2_img)
    # transform = transforms.ToTensor()
    # tensor = transform(cv2_img).permute(2,0,1)
    tensor = torch.tensor(cv2_img, dtype=torch.float).permute(2, 0, 1)
    # tensor = tensor.reshape(1, 3, 101, 101)  # MAY BE SOURCE OF ERROR
    tensor = torch.unsqueeze(tensor, dim=0)
    # print(tensor)
    # cv2.imwrite('camera_image.jpeg', cv2_img)
    # Testing the Model
    prediction = model(tensor)
    ##print(prediction[0])
    prediction = prediction.detach().cpu().numpy()  # OK
    prediction = prediction[0]
    left = prediction[0]
    straight = prediction[1]
    right = prediction[2]
    left_arr.append(left)
    left_arr.pop(0)
    straight_arr.append(straight)
    straight_arr.pop(0)
    right_arr.append(right)
    right_arr.pop(0)
   
    left = sum(left_arr)/len(left_arr)
    straight = sum(straight_arr)/len(straight_arr)
    right = sum(right_arr)/len(right_arr)
    
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
    
    # print("Left: ", left, ", straight: ", straight, ", right: ", right, " .")
    print(f'{left:.4f}',f'{straight:.4f}', f'{right:.4f}', end = '\r')

    speed_straight = 0.5
    speed_angular = 0.05
    message = VelocityReferenceStamped()


    message.reference.use_heading_rate = 1
    message.header.frame_id = uav[1:] + "fcu"
    rate = (prediction[2]-prediction[0]) * speed_angular
    message.reference.heading_rate = rate
    ##print("Rate is ", rate)
    #if prediction[2]-prediction[0] < 0.3:
    message.reference.velocity.x = speed_straight*prediction[1]
    # else:
    #     message.reference.velocity.x = speed_straight * prediction[1] * 10


    pub.publish(message)

heading = None

def image_callback(img_msg):
    image = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    compute(image, heading)

    # for i in range(1, 10000000):
    #     pass

def heading_callback(hdg_msg):
    global heading
    heading = hdg_msg



'''
def callback(img_msg, heading_msg):
    # print("Received an image!")
        # Convert your ROS Image message to OpenCV2
    cv2_img = bridge.imgmsg_to_cv2(img_msg, "rgb8")
    # cv2_img = cv2.imread("test.jpg")
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.resize(cv2_img, (101, 101), interpolation = cv2.INTER_AREA)
    cv2_img = np.asarray(cv2_img)
    # transform = transforms.ToTensor()
    # tensor = transform(cv2_img).permute(2,0,1)
    tensor = torch.tensor(cv2_img, dtype=torch.float).permute(2,0,1)
    # tensor = tensor.reshape(1, 3, 101, 101)  # MAY BE SOURCE OF ERROR
    tensor = torch.unsqueeze(tensor, dim=0)
    #print(tensor)
    # cv2.imwrite('camera_image.jpeg', cv2_img)
    # Testing the Model
    prediction = model(tensor)
    print(prediction[0])
    prediction = prediction.detach().cpu().numpy()  # OK
    prediction = prediction[0]
    left = prediction[0]
    straight = prediction[1]
    right = prediction[2]
    
    #print("Left: ", left, ", straight: ", straight, ", right: ", right, " .")
    print(prediction)

    message = VelocityReferenceStamped()
    message.reference.velocity.z = 0
    pub.publish(message)
'''

''' Model '''

class Simple_Conv_Model(torch.nn.Module):
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
        #self.tanh = torch.nn.Tanh()
        self.weight_init()

    def forward(self, x):
        #save_image(x[0]/255, 'imtest.jpg')
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


#Hyperparameters
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
    image_topic = "/camera/realsense2_camera/color/image_raw"
    heading_topic = uav + "control_manager/position_cmd"
    # Set up your subscriber and define its callback
    # rospy.Subscriber(image_topic, Image, image_callback)
    '''
    image_sub = rospy.Subscriber(image_topic, Image)
    heading_sub = rospy.Subscriber(heading_topic, Heading)

    ts = message_filters.TimeSynchronizer([image_sub, heading_sub], 10)
    ts.registerCallback(callback) '''

    # Spin until ctrl + c
    #cv2.imwrite('camera.jpeg', Image)
    # rospy.Subscriber(heading_topic, Float64Stamped, heading_callback)
    rospy.Subscriber(image_topic, Image, image_callback)
    print("Subsciber initialized")
    rospy.spin()

if __name__ == '__main__':
    print("Path follower V1.0")
    #Modules
    model = Simple_Conv_Model(nbr_classes=nbr_cls)
    model.load_state_dict(torch.load(os.path.abspath(os.path.dirname(__file__)) + '/weights_nl_final.pth', map_location='cpu'))
    np.set_printoptions(suppress=True)
    main()
