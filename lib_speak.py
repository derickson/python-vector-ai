import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define a function to speak text
def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()