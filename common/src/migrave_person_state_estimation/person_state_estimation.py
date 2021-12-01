import os

import joblib
import numpy as np


class PersonStateEstimation(object):
    """
    Person state estimation. Engagement is estimated using xgboost
    """

    def __init__(self, config, pkg_dir):
        self._config = config
        if "engagement" in self._config:
            cls_path = os.path.join(pkg_dir,
                                    "common",
                                    "models",
                                    self._config["engagement"]["model_file"])

            self._engagement_cls, self._engagement_mean, self._engagement_std = self.load_classifier(cls_path)

    def load_classifier(self, cls_path):
        with open(cls_path, 'rb') as f:
            classifier, mean, std = joblib.load(f)

        return classifier, mean, std

    def estimate_engagement(self, features, normalize=True):
        """
        Estimate engagement given the current face features, audio signal,
        and game performance and return engagement scores

        :param features:       Face, audio, or game performance features
        :type name:            numpy.array

        :return:               Engagement
        """

        if normalize:
            features -= np.array(self._engagement_mean)
            features /= self._engagement_std
        
        probabilities = self._engagement_cls.predict_proba([features])[0]
        max_index = np.argmax(probabilities)
        prediction = self._engagement_cls.classes_[max_index]

        return prediction, probabilities[max_index]
