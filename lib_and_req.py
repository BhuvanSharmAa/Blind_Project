import cv2
import numpy as np
import threading
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from ultralytics import YOLO
from multiprocessing import Process, Queue, Value
import time
import os

if not os.path.exists("coco.names"):
    import urllib.request

    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    urllib.request.urlretrieve(url, "coco.names")
model = YOLO('yolov8n.pt')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
engine = pyttsx3.init()
   
