# migrave_person_state_estimation
Repository for a person state estimation component used in the MigrAVE project.

### Requirements:
```
xgboost == 1.5.0
sklearn == 0.24.2
joblib == 1.0.1
```

### Package dependencies
Person state estimation requires [`migrave_face_feature_detector`](https://github.com/migrave/migrave_perception/tree/main/migrave_face_feature_detector) which computes face features using `openface`.
  ```
  roslaunch migrave_face_feature_detector face_feature_detector.launch
  ```

### Usage
* Launch the node:
  
  ```
  roslaunch migrave_person_state_estimation person_state_estimation.launch
  ```

* Start engagement estimation by sending an event to the following topic:

  ```
  rostopic pub /migrave_perception/person_state_estimator/event_in std_msgs/String e_start
  ```

* Stop engagement estimation

  ```
  rostopic pub /migrave_perception/person_state_estimator/event_in std_msgs/String e_stop
  ```

* Outputs
  The node publishes [`AffectiveState`](https://github.com/migrave/migrave_ros_msgs/blob/main/person_state/AffectiveState.msg) message to the following topic:
  ```
  /migrave_perception/person_state_estimator/affective_state
  ```
