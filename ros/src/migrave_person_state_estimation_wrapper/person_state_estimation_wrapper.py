import os
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge

from rospkg import RosPack
from sensor_msgs.msg import Image
from std_msgs.msg import String
from actionlib import SimpleActionServer

from migrave_common import file_utils
from migrave_ros_msgs.msg import (AffectiveState, AudioFeatures, Faces,
                                  OverallGamePerformance, Person,
                                  GetAverageEngagementAction,
                                  GetAverageEngagementGoal,
                                  GetAverageEngagementResult)
from migrave_person_state_estimation.person_state_estimation import \
    PersonStateEstimation

MIGRAVE_AU_NAMES = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06',
                    'AU07', 'AU09', 'AU10', 'AU12', 'AU14',
                    'AU15', 'AU17', 'AU20', 'AU23', 'AU25',
                    'AU26', 'AU28', 'AU45']


class PersonStateEstimationWrapper(object):

    def __init__(self):
        self._config_file = rospy.get_param("~config_path", None)
        pkg_path = RosPack().get_path("migrave_person_state_estimation")

        if not os.path.isfile(self._config_file):
            rospy.logwarn("Config file is not given or does not exist")

        self._config = file_utils.parse_yaml_config(self._config_file)
        self._debug = rospy.get_param("~debug", False)

        self._face_feature_topic = rospy.get_param("~face_feature_topic", "/face_features")
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
        self._sub_audio_features = rospy.Subscriber(self._audio_feature_topic, AudioFeatures, self.audio_feature_cb)

        self._cvbridge = CvBridge()

        self._engagement_cache_duration_seconds = rospy.get_param("~engagement_cache_seconds", 5.)
        self._engagement_estimate_cache = {}

        self._avg_engagement_estimation_server_name = rospy.get_param("~avg_engagement_estimation_server_name",
                                                                      "/migrave/get_avg_engagement")
        self._avg_engagement_estimation_server = SimpleActionServer(self._avg_engagement_estimation_server_name,
                                                                    GetAverageEngagementAction,
                                                                    self.get_avg_engagement)

    def face_feature_cb(self, data: Faces) -> None:
        rospy.logdebug("Face feature msg received")
        for i, face in enumerate(data.faces):
            # we inititialise an engagement queue for the current
            # user if one doesn't already exits
            if i not in self._engagement_estimate_cache:
                self._engagement_estimate_cache[i] = deque()

            # we retrieve features required for engagement estimation
            auc_dict = {}
            for ac in face.action_units:
                auc_dict[ac.name] = ac.presence

            face_features = []
            for au_name in MIGRAVE_AU_NAMES:
                # we format the name of the AU feature as expected by the classifier
                au_feature_name = f'of_{au_name}_c'
                face_features.append((au_feature_name, auc_dict[au_name]))

            # left gaze: of_gaze_0_x, ..y, ..z
            face_features.append(('of_gaze_0_x', face.left_gaze.position.x))
            face_features.append(('of_gaze_0_y', face.left_gaze.position.y))
            face_features.append(('of_gaze_0_z', face.left_gaze.position.z))

            # right gaze: of_gaze_1_x, ..y, ..z
            face_features.append(('of_gaze_1_x', face.right_gaze.position.x))
            face_features.append(('of_gaze_1_y', face.right_gaze.position.y))
            face_features.append(('of_gaze_1_z', face.right_gaze.position.z))

            # gaze angle
            face_features.append(('of_gaze_angle_x', face.gaze_angle.x))
            face_features.append(('of_gaze_angle_y', face.gaze_angle.y))

            # head pose
            face_features.append(('of_pose_Tx', face.head_pose.position.x))
            face_features.append(('of_pose_Ty', face.head_pose.position.y))
            face_features.append(('of_pose_Tz', face.head_pose.position.z))
            face_features.append(('of_pose_Rx', face.head_pose.orientation.x))
            face_features.append(('of_pose_Ry', face.head_pose.orientation.y))
            face_features.append(('of_pose_Rz', face.head_pose.orientation.z))

            # estimate engagement
            engagement_score, prob = self._pse.estimate_engagement(face_features)
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

            # we update the engagement estimate cache for the current user
            engagement_estimate_time = rospy.Time.now().to_sec()
            if self._engagement_estimate_cache[i]:
                # we clean old entries in the cache (older than the maximum cache duration)
                clean_cache = True
                while clean_cache:
                    time_diff_to_first_estimate = (engagement_estimate_time - self._engagement_estimate_cache[i][0][0])
                    if time_diff_to_first_estimate > self._engagement_cache_duration_seconds:
                        self._engagement_estimate_cache[i].popleft()

                        # we stop if the cache is empty after removing the element removal
                        if not self._engagement_estimate_cache[i]:
                            clean_cache = False
                    else:
                        clean_cache = False
            self._engagement_estimate_cache[i].append((engagement_estimate_time,
                                                       engagement_score))

            # we publish a debug image for visualising the engagement score
            if self._debug:
                rospy.logdebug("Engagement %f", engagement_score)
                cv_image = self._cvbridge.imgmsg_to_cv2(face.debug_image, "bgr8")
                image = cv2.putText(cv_image,
                                    'Engaged' if engagement_score >= 1.0 else "Disengaged",
                                    (100, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    (0, 255, 0) if engagement_score >= 1.0 else (0, 0, 255),
                                    2,
                                    cv2.LINE_AA)
                image = np.array(image, dtype=np.uint8)
                cv_image_debug = self._cvbridge.cv2_to_imgmsg(image, "bgr8")
                self._pub_debug_img.publish(cv_image_debug)

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

    def get_avg_engagement(self, goal: GetAverageEngagementGoal) -> None:
        result = GetAverageEngagementResult()

        # ideally, the user ID should be passed on with the message, but we now
        # assume that we only have one user that can be seen; we return an empty
        # result if no engagement estimates have been made thus far
        if 0 not in self._engagement_estimate_cache or not self._engagement_estimate_cache[0]:
            rospy.logwarn(f"[get_avg_engagement] No engagement estimates for average calculation")
            self._avg_engagement_estimation_server.set_succeeded(result)
            return

        # we create a copy of the cache so that we work with a "frozen" cache version
        # rather than one that might be changed while the cache is being read
        raw_engagement_estimates = list(self._engagement_estimate_cache[0])
        estimates_within_time_range = [(t, e) for (t, e) in raw_engagement_estimates
                                       if goal.start_time <= t <= goal.end_time]

        # we return an empty result if there are no engagement estimates
        # within the specified time range
        if not estimates_within_time_range:
            rospy.logwarn(f"[get_avg_engagement] No estimates within range {goal.start_time} - {goal.end_time}")
            self._avg_engagement_estimation_server.set_succeeded(result)
            return

        # we estimate the average engagement for each second within the given time range
        rospy.loginfo(f"[get_avg_engagement] Calculating average engagement values between {goal.start_time} - {goal.end_time}")
        estimate_idx = 0
        current_first_estimate_idx = 0
        while estimate_idx < len(estimates_within_time_range):
            time_diff = estimates_within_time_range[estimate_idx][0] - estimates_within_time_range[current_first_estimate_idx][0]
            if time_diff > 1:
                s = sum([e for _, e in estimates_within_time_range[current_first_estimate_idx:estimate_idx]])
                avg = s / (estimate_idx - current_first_estimate_idx)

                result.timestamps.append(estimates_within_time_range[estimate_idx-1][0])
                result.avg_engagement.append(avg)
                current_first_estimate_idx = estimate_idx
            estimate_idx += 1

        rospy.loginfo("[get_avg_engagement] Done calculating; setting result")
        self._avg_engagement_estimation_server.set_succeeded(result)

    def initialize(self) -> None:
        #start event in and out
        self._sub_event = rospy.Subscriber("~event_in", String, self.event_callback)
        self._pub_event = rospy.Publisher("~event_out", String, queue_size=1)
