
class PersonStateEstimation(object):
    """
    Person state estimation
    """

    def __init__(self, config):
        self._config = config

    def get_state_estimate(self, image=None, game_performance=None, audio=None, skeletons=None):
        """
        Estimate person state given the current image (face, body), audio signal,
        and game performance and return valence, arousal, and engagement scores

        :param image:       RGB image
        :type name:         numpy.array

        :param image:       Audio signal
        :type name:         numpy.array

        :param image:       Game performance
        :type name:         numpy.float64

        :return:            valence, arousal, engagement
        """

        valence = 0.0
        arousal = 0.0
        engagement = 0.0

        # if audio:
        #     audio_features = self._audio_feature_extractor.get_audio_features(audio)

        # if game_performance:
        #     face_features = self._face_feature_extractor.get_face_features(image)

        # detected_gaze = self._gaze_extractor.get_gaze(image)

        # if skeletons is None:
        #     skeletons = self._skeleton_tracking.get_skeletons(image)

        #Compute valence, arousal and engament based on audio, face,
        #gaze and skeleton features.

        return valence, arousal, engagement
        
