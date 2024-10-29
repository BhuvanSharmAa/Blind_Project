import cv2
import numpy as np
import threading
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from ultralytics import YOLO
from multiprocessing import Process, Queue, Value
import time


model = YOLO('yolov8n.pt')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
engine = pyttsx3.init()
detected_objects = []
frame_queue = Queue(maxsize=5)
command_processing = Value('b', False)  
collision_alerted = Value('b', False)    
