import os
import threading
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from std_msgs.msg import String
from migrave_ros_msgs.msg import AffectiveState, AudioFeatures, Face, OverallGamePerformance, Person
from qt_nuitrack_app.msg import Faces, FaceInfo

from migrave_person_state_estimation.person_state_estimation import PersonStateEstimation
from migrave_person_state_estimation_wrapper import utils


class PersonStateEstimationWrapper(object):

    def __init__(self):
        self.lock = threading.Lock()
        self._config_file = rospy.get_param("~config_path", None)

        if not self._config_file or os.path.isfile(self._config_file):
            rospy.logwarn("Config file is not given or does not exist")

        self._config = utils.parse_yaml_config(self._config_file)
        self._debug = rospy.get_param("~debug", False) 
        self._demo = rospy.get_param("~demo", False) 
        # Time from which only messages from face_msg_time_to_keep until now() will be considered
        self._face_msg_time_to_keep = rospy.get_param("~face_msg_time_to_keep", 1.0) 

        self._face_feature_topic = rospy.get_param("~face_feature_topic", "/face_features") 
        self._audio_feature_topic = rospy.get_param("~audio_feature_topic", "/audio_features") 
        self._skeleton_topic = rospy.get_param("~skeleton_topic", "/skeletons") 

        # publishers
        self._pub_affective_state = rospy.Publisher("~affective_state", AffectiveState, queue_size=1)
        
        # subscribers
        # self._sub_face_features = rospy.Subscriber(self._face_feature_topic, Face, self.face_feature_cb)
        # self._sub_audio_features = rospy.Subscriber(self._audio_feature_topic, AudioFeatures, self.audio_feature_cb)
        # self._sub_skeletons = rospy.Subscriber(self._skeleton_topic, self.skeleton_cb, queue_size=1)

        self._cvbridge = CvBridge()
        self._person_state_estimator = PersonStateEstimation(self._config)

        self.nuitrack_face_features = None
        self.nuitrack_face_feature_time = None

    def face_feature_cb(self, data: Face) -> None:
        rospy.logdebug("Face feature msg received")

    def nuitrack_face_feature_cb(self, data):
        self.lock.acquire()
        self.nuitrack_face_features = data.faces
        self.nuitrack_face_feature_time = rospy.Time.now()
        self.lock.release()
    
    def preprocess_nuitrack_face_and_publish(self) -> None:
        self.lock.acquire()
        nuitrack_face_features = self.nuitrack_face_features
        nuitrack_face_time = self.nuitrack_face_feature_time
        self.lock.release()

        if nuitrack_face_features and (rospy.Time.now() - nuitrack_face_time) < \
            rospy.Duration(self._face_msg_time_to_keep):
            emotions = ["neutral", "angry", "happy", "surprise"]
            for nt_face_feature in nuitrack_face_features:
                affective_state = AffectiveState()
                affective_state.stamp = rospy.Time.now()

                person = Person()
                person.id = str(nt_face_feature.id)
                person.name = "MigrAVE"
                person.age = nt_face_feature.age_years
                person.gender = nt_face_feature.gender
                person.mother_tongue = "German"
                affective_state.person = person

                affective_state.valence = 0.0
                affective_state.arousal = 0.0
                affective_state.engagement = 0.0

                max_emotion_prob_idx = np.argmax(np.asarray([nt_face_feature.emotion_neutral,
                                                             nt_face_feature.emotion_angry,
                                                             nt_face_feature.emotion_happy,
                                                             nt_face_feature.emotion_surprise]))
                affective_state.emotion = emotions[max_emotion_prob_idx]

                self._pub_affective_state.publish(affective_state)

            # self._sub_nuitrack_face_feature.unregister()

    def audio_feature_cb(self, data: AudioFeatures) -> None:
       rospy.logdebug("Audio feature msg received")

    # def skeleton_cb(self, data: SkeletonMsg) -> None:
    #    rospy.logdebug("Skeleton msg received")

    def estimate_person_affective_state(self) -> None:
        #estimate affective state
        rospy.logdebug("Start estimating person's affective state")
        person_state_estimate = self._person_state_estimator.get_state_estimate()

        affective_state = AffectiveState()
        affective_state.stamp = rospy.Time.now()
        affective_state.valence = person_state_estimate[0]
        affective_state.arousal = person_state_estimate[1]
        affective_state.engagement = person_state_estimate[2]

        # ToDo: add person
        person = Person()
        person.id = ""
        person.name = "unknown"
        person.age = 5
        person.gender = "unidentified"
        person.mother_tongue = "German"
        affective_state.person = person

        self._pub_affective_state.publish(affective_state)
        
    def event_callback(self, data: String) -> None:
        event_out_data = String()
        if data.data == "e_start":
            # subscribe to face features (either from nuitrack or openface)
            if "qt_nuitrack_app" in self._face_feature_topic:
                self._sub_nuitrack_face_feature = rospy.Subscriber(self._face_feature_topic, Faces, self.nuitrack_face_feature_cb)
                self.preprocess_nuitrack_face_and_publish()
            else:
                self._sub_face_feature = rospy.Subscriber(self._face_feature_topic, Face, self.face_feature_cb)

            #ToDo
            #Subscribe to gaze, face features, audio features, and game performance

            #estimate person affective state given the above features
            # self.estimate_person_affective_state()

            event_out_data.data = "e_started"
            self._pub_event.publish(event_out_data)

        elif data.data == "e_stop":
            self._sub_nuitrack_face_feature.unregister()
            event_out_data.data = "e_stopped"
            self._pub_event.publish(event_out_data)

    def run(self) -> None:
        #start event in and out
        self._sub_event = rospy.Subscriber("~event_in", String, self.event_callback)
        self._pub_event = rospy.Publisher("~event_out", String, queue_size=1)

