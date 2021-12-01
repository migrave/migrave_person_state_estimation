import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from migrave_common import file_utils
from migrave_person_state_estimation.person_state_estimation import \
    PersonStateEstimation
from migrave_ros_msgs.msg import (AffectiveState, AudioFeatures, Faces,
                                  OverallGamePerformance, Person)
from rospkg import RosPack
from sensor_msgs.msg import Image
from std_msgs.msg import String


class PersonStateEstimationWrapper(object):

    def __init__(self):
        self._config_file = rospy.get_param("~config_path", None)
        pkg_path = RosPack().get_path("migrave_person_state_estimation")

        if not os.path.isfile(self._config_file):
            rospy.logwarn("Config file is not given or does not exist")

        self._config = file_utils.parse_yaml_config(self._config_file)
        self._debug = rospy.get_param("~debug", False) 

        self._face_feature_topic = rospy.get_param("~face_feature_topic", "/face_features") 
        self._of_debug_img_topic = rospy.get_param("~of_debug_img_topic", "/of_debug_img_topic") 
        self._audio_feature_topic = rospy.get_param("~audio_feature_topic", "/audio_features") 
        self._skeleton_topic = rospy.get_param("~skeleton_topic", "/skeletons") 

        # Person state estimation
        self._pse = PersonStateEstimation(self._config, pkg_path)

        self.nuitrack_face_features = None
        self.nuitrack_face_feature_time = None

        # publishers
        self._pub_affective_state = rospy.Publisher("~affective_state", AffectiveState, queue_size=1)
        self._pub_debug_img = rospy.Publisher("~debug_image", Image, queue_size=1)
        
        # subscribers
        self._sub_face_features = rospy.Subscriber(self._face_feature_topic, Faces, self.face_feature_cb)
        self._sub_of_debug_img = rospy.Subscriber(self._of_debug_img_topic, Image, self.debug_img_cb)
        self._sub_audio_features = rospy.Subscriber(self._audio_feature_topic, AudioFeatures, self.audio_feature_cb)

        # save global engagement score for  debugging
        self._engagement_score = None
        self._engagement_prob = None

        self._cvbridge = CvBridge()

    def face_feature_cb(self, data: Faces) -> None:
        rospy.logdebug("Face feature msg received")
        for i,face in enumerate(data.faces):
            face_features = []
            # get auc
            auc_dict = {}
            for ac in face.action_units:
                auc_dict[ac.name] = ac.presence
            # Sort auc according to the dataset
            auc_dict = dict(sorted(auc_dict.items()))
            face_features.extend(auc_dict.values())

            # left gaze: of_gaze_0_x, ..y, ..z
            face_features.append(face.left_gaze.position.x)
            face_features.append(face.left_gaze.position.y)
            face_features.append(face.left_gaze.position.z)

            # right gaze: of_gaze_0_x, ..y, ..z
            face_features.append(face.right_gaze.position.x)
            face_features.append(face.right_gaze.position.y)
            face_features.append(face.right_gaze.position.z)

            # gaze angle
            face_features.append(face.gaze_angle.x)
            face_features.append(face.gaze_angle.y)

            # head pose
            face_features.append(face.head_pose.position.x)
            face_features.append(face.head_pose.position.y)
            face_features.append(face.head_pose.position.z)
            face_features.append(face.head_pose.orientation.x)
            face_features.append(face.head_pose.orientation.y)
            face_features.append(face.head_pose.orientation.z)

            # estimate engagement
            engagement_score, prob = self._pse.estimate_engagement(np.asarray(face_features))
            rospy.logdebug("Engagement score = %f, engagement prob = %f", engagement_score, prob)

            affective_state = AffectiveState()
            affective_state.stamp = rospy.Time.now()

            # ToDo: add estimation for valence and arousal
            affective_state.valence = 0.0
            affective_state.arousal = 0.0

            affective_state.engagement = engagement_score

            # ToDo: add person
            person = Person()
            affective_state.person.id = str(i)

            # Publish affective_state
            self._pub_affective_state.publish(affective_state)

            if self._debug:
                self._engagement_score = engagement_score
                self._engagement_prob = prob

    def debug_img_cb(self, data: Image) -> None:
        rospy.logdebug("Engagement %f", self._engagement_score)
        if self._engagement_score:
            cv_image = self._cvbridge.imgmsg_to_cv2(data, "bgr8")
            image = cv2.putText(cv_image,
                                'Engaged' if self._engagement_score >= 1.0 else "Disengaged",
                                (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0) if self._engagement_score >= 1.0 else (0, 0, 255),
                                2,
                                cv2.LINE_AA)
            image = np.array(image, dtype=np.uint8)
            cv_image_debug = self._cvbridge.cv2_to_imgmsg(image, "bgr8")
            self._pub_debug_img.publish(cv_image_debug)
            self._engagement_score = None

    def audio_feature_cb(self, data: AudioFeatures) -> None:
       rospy.logdebug("Audio feature msg received")
        
    def event_callback(self, data: String) -> None:
        event_out_data = String()
        if data.data == "e_start":
            # subscribe to face features
            self._sub_face_feature = rospy.Subscriber(self._face_feature_topic,
                                                      Faces,
                                                      self.face_feature_cb)

            event_out_data.data = "e_started"
            self._pub_event.publish(event_out_data)

        elif data.data == "e_stop":
            self._sub_face_feature.unregister()
            event_out_data.data = "e_stopped"
            self._pub_event.publish(event_out_data)

    def initialize(self) -> None:
        #start event in and out
        self._sub_event = rospy.Subscriber("~event_in", String, self.event_callback)
        self._pub_event = rospy.Publisher("~event_out", String, queue_size=1)

