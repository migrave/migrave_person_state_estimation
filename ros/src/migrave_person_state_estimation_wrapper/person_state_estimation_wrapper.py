import os
import rospy
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from std_msgs.msg import String
from migrave_ros_msgs.msg import AffectiveState, AudioFeatures, Face, OverallGamePerformance, Person

from migrave_person_state_estimation.person_state_estimation import PersonStateEstimation
from migrave_person_state_estimation_wrapper import utils


class PersonStateEstimationWrapper(object):

    def __init__(self):
        self._config_file = rospy.get_param("~config_path", None)

        if not self._config_file or os.path.isfile(self._config_file):
            rospy.logerr("Config file is not given or does not exist")

        self._config = utils.parse_yaml_config(self._config_file)
        self._debug = rospy.get_param("~debug", False) 
        self._demo = rospy.get_param("~demo", False) 

        self._face_feature_topic = rospy.get_param("~face_feature_topic", "/face_features") 
        self._audio_feature_topic = rospy.get_param("~audio_feature_topic", "/audio_features") 
        self._skeleton_topic = rospy.get_param("~skeleton_topic", "/skeletons") 
        self._game_performance_topic = rospy.get_param("~game_performance", "/game_performance") 

        # publishers
        self._pub_affective_state = rospy.Publisher("~affective_state", AffectiveState, queue_size=1)
        
        # subscribers
        self._sub_face_features = rospy.Subscriber(self._face_feature_topic, Face, self.face_feature_cb)
        self._sub_audio_features = rospy.Subscriber(self._audio_feature_topic, AudioFeatures, self.audio_feature_cb)
        # self._sub_skeletons = rospy.Subscriber(self._skeleton_topic, self.skeleton_cb, queue_size=1)
        self._sub_game_performance = rospy.Subscriber(self._game_performance_topic, OverallGamePerformance, self.game_performance_cb)

        self._cvbridge = CvBridge()
        self._person_state_estimator = PersonStateEstimation(self._config)

    def face_feature_cb(self, data: Face) -> None:
       rospy.logdebug("Face feature msg received")
    
    def audio_feature_cb(self, data: AudioFeatures) -> None:
       rospy.logdebug("Audio feature msg received")

    # def skeleton_cb(self, data: Face) -> None:
    #    rospy.logdebug("Skeleton msg received")

    def game_performance_cb(self, data: OverallGamePerformance) -> None:
       rospy.logdebug("Overal game performance msg received")

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
            #ToDo
            #Subscribe to gaze, face features, audio features, and game performance

            #estimate person affective state given the above features
            self.estimate_person_affective_state()

            event_out_data.data = "e_started"
            self._pub_event.publish(event_out_data)

        elif data.data == "e_stop":
            event_out_data.data = "e_stopped"
            self._pub_event.publish(event_out_data)

    def run(self) -> None:
        #start event in and out
        self._sub_event = rospy.Subscriber("~event_in", String, self.event_callback)
        self._pub_event = rospy.Publisher("~event_out", String, queue_size=1)
