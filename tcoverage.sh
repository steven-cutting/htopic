#! /usr/bin/env bash


env coverage run -a --source=h_topic_model $(which py.test)
env coverage report -m
