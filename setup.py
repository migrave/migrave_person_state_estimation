#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['migrave_person_state_estimation',
              'migrave_person_state_estimation_wrapper'],
    package_dir={'migrave_person_state_estimation': 'common/src/migrave_person_state_estimation',
                 'migrave_person_state_estimation_wrapper': 'ros/src/migrave_person_state_estimation_wrapper'}
)

setup(**d)
