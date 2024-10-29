nlp = spacy.load("en_core_web_sm")  # Named Entity Recognition
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def speak(text):
    engine.say(text)
    engine.runAndWait()


def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()


def get_speech_input(callback):
    global command_processing
    if command_processing:
        return
    command_processing = True

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        speech_text = recognizer.recognize_google(audio)
        print(f"You said: {speech_text}")
        callback(speech_text.lower())
    except sr.UnknownValueError:
        print("Could not understand audio.")
        callback("")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        callback("")
    finally:
        command_processing = False
