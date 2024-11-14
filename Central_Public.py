# CENTRAL 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸
# Go to https://huggingface.co/settings/tokens and get a token then paste inside API_TOKEN = ""
# Copy paste the following code line in your terminal below:
# pip install SpeechRecognition beautifulsoup4 sounddevice gTTS numpy pyaudio
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from requests.exceptions import ConnectionError
import speech_recognition as sr
from bs4 import BeautifulSoup
import sounddevice as sd
from PIL import Image
from gtts import gTTS
import numpy as np
import webbrowser
import threading
import requests
import random
import time
import mss
import cv2
import os

from flask import Flask
import webbrowser
import threading






API_TOKEN = "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" # Insert your token from Hugging Face in between the " "
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
central_ai_speaking = False
previous_generated_text = ""
voicebox_timeout = 20
timeout_duration = 15
phrase_counts = {}
global your_statement
your_statement = ""
oscillation_duration = 0.5
technology_count = 0


technology_terminologiesa = [
'Swarru of Erra',
'Boston Dynamics',
'AIBO',
'Acoustic levitation',
'Action at a distance',
'Actor observer bias',
'Ad hominem',
'Aerodynamics',
'Agree to disagree',
'Anchoring effect',
'Anyon',
'Appeal to tradition',
'Aquifer',
'Argument from ignorance',
'Artificial womb',
'Atom',
'Attribution bias',
'Authority bias',
'Availability bias',
'BCI2000',
'Balkanization',
'Barnum effect',
'Base 12 Math',
'Base rate bias',
'Belief perseverance',
'Biefeld–Brown effect',
'Bioelectronics',
'Bioluminescent bacteria',
'Bragging',
'Brainwave',
'Brainwave entrainment',
'CERN',
'Casimir effect',
'Cathode',
'Cell Nucleus',
'Centrifugal Acceleration',
'Centrifugal force',
'Cherry picking data',
'Chirality (physics)',
'Circular polarization',
'Cognitive Dissonance',
'Cold fusion',
'Confirmation bias',
'Controversy',
'Convolutional neural networks',
'Coriolis force',
'Crop Circle Mathematics',
'Crystal',
'Curse of knowledge',
'Cymatics',
'Buoyant levitation',
'Delayed gratification',
'Diamagnetic levitation',
'Diamagnetism',
'Dielectric',
'Diffusion Limited Aggregation',
'Dolphin',
'Double standard',
'Drag (physics)',
'Dunning-Kruger Effect',
'E8 (mathematics)',
'Earth battery',
'Ecocapsule',
'Ecocapsule',
'Edwin Gray Electric Motor',
'Electric Field',
'Electrical resistivity and conductivity',
'Electroencephalography',
'Electrogravitics',
'Electrogravity',
'Electromagnetic levitation',
'Electron capture',
'Electronegativity',
'Electronic skin',
'Electrostatic',
'Electrostatic generator',
'Electrostatic levitation',
'Epicyclic gearing',
'Epigenetics',
'Ernst Chladni',
'Ethnocentrism',
'Eugene Mallove',
'Evolutionary programming',
'Exoatmospheric Kill Vehicle',
'Expectancy theory',
'Factorial',
'False advertising',
'False consensus effect',
'Faulty generalization',
'Fearmongering',
'Fermats spiral',
'Fermentation',
'Ferrite (magnet)',
'Ferrofluid',
'Fibonacci Sequence',
'Filler (linguistics)',
'Fluid dynamics',
'Fractal',
'Fractal-generating software',
'Free Energy Device',
'Free range',
'Functional fixedness',
'Fundamental Attribution Error',
'Gaia, Inc.',
'Gamblers Fallacy',
'Gamification',
'Garbage in, garbage out',
'Gas centrifuge',
'Gaslighting',
'Gausss law for gravity',
'Generalization',
'Generative Adversarial Network',
'Generator disks',
'Glider (aircraft)',
'Glymphatic system',
'Gravifugal Force',
'Gravitational Field',
'Gravitational Wave',
'Gravitomagnetism',
'Groupthink',
'Guilt trip',
'Gut microbiota',
'Gyrocompass',
'Gyroscope',
'Hindsight bias',
'Histamine',
'Holography',
'Homopolar generator',
'Homopolar motor',
'Paramagnetism',
'Human vestigiality',
'Humane society',
'Hwang Woo-suk',
'IKEA effect',
'Illusory superiority',
'Immune network theory',
'In-group favoritism',
'Induction motor',
'Intellectual humility',
'Interferometers',
'Internship',
'Van Allen radiation belt',
'Intestinal permeability',
'Ionic conductivity (solid state)',
'Ketones',
'Kuiper Belt',
'Laithwaite Engine',
'Law of Mirrors',
'Lectin',
'Lift (force)',
'Lift-induced drag',
'Liquid crystal',
'List of cognitive biases',
'List of particles',
'Parthenogenesis',
'Load factor (aeronautics)',
'Logarithmic Spiral',
'Lorentz ether theory',
'Luminiferous aether',
'MNIST database',
'Mach number',
'Magnet',
'Magnet wire',
'Magnetic Field',
'Magnetic levitation',
'Magnifying Transmitter',
'Magnus effect',
'Mandelbrot Set',
'Mass flow rate',
'Mathematical and theoretical biology',
'Media bias',
'Mendocino motor',
'Mere Exposure Effect',
'Methernitha',
'Microbiome',
'Microorganism',
'Miniaturization',
'Mobbing',
'Mucus membranes',
'Nano Accelerator of Particles',
'Nanotechnology',
'Negativity bias',
'Neural oscillation',
'Newmans energy machine',
'Newtons law of universal gravitation',
'Nikola Tesla',
'Normative statement',
'Not invented here',
'Nuclear Fusion',
'Nuclear reaction',
'Nuclear transfer',
'Obstruction of justice',
'One-upmanship',
'Operant conditioning',
'Optical levitation',
'Au pair program',
'Oscillator',
'Outcome bias',
'Overconfidence bias',
'Pareidolia',
'Passive-aggressive behavior',
'Patterns in nature',
'People for the Ethical Treatment of Animals',
'Permanent magnet motor',
'Photovoltaics',
'Phyllotaxis',
'Plant cognition',
'Plasma actuator',
'Platelet',
'Platelet Rich Plasma',
'Platonic Solid',
'Pleiades',
'Polarization (waves)',
'Polarization tensor',
'Present bias',
'Procrastination',
'Psilocybin',
'Quantum entanglement',
'Quantum mechanics',
'Quasi Crystals',
'Quasiparticle',
'Rare-earth magnet',
'Rationalization (psychology)',
'Recency bias',
'Reductio ad absurdum',
'Regenerative Medicine',
'Reinforcement learning',
'Resonant Frequency',
'Reticulum',
'Reversed field pinch',
'Rodin coil',
'Rotating magnetic field',
'Rumination (psychology)',
'Sacred Geometry',
'Scalar field theory',
'Scenar Therapy',
'SciPy',
'Searle Disk',
'Selective omission',
'Self serving bias',
'Self-similarity',
'Semmelweis reflex',
'Simple Magnetic Overunity Toy',
'Superconducting Levitation',
'Social-desirability bias',
'Solenoid',
'Somatic Cell',
'Somatic Cell Nuclear Transfer',
'Special pleading',
'Speciesism',
'Spermatozoon',
'Spotlight effect',
'Stan Meyer Water Fuel Car',
'Starship Coil',
'Status quo bias',
'Stem cell',
'Straw man',
'Subjectivity and objectivity (philosophy)',
'Subliminal stimuli',
'Sunk cost',
'Superconductivity',
'Survivorship bias',
'Sympathetic Resonance',
'Synaptic pruning',
'Tachyon',
'Tachyonic field',
'Tacit assumption',
'Tagyeta',
'Taxonomy (biology)',
'Telautomatics',
'Telomeres',
'Tensorflow',
'Tesla Coil',
'Testatika',
'Theta wave',
'Thomas Townsend Brown',
'Thrust vectoring',
'Toroidal and poloidal coordinates',
'Torsion (mechanics)',
'Torus',
'Tractor beam',
'Tri-arcuate ballistic electrode',
'Triode',
'Unified Field Quadrangle',
'Vacuum',
'Variable-frequency drive',
'Nutri-Score',
'Vierordts law',
'Von Restorff Effect',
'Vortex ring',
'Wardenclyffe Tower',
'Water power engine',
'Waves in plasmas',
'Wishful thinking',
'World Wireless System',
'Deep Underground Military Bases',
'Xenophobia',
'Zeigarnik effect',
'Zeta Reticuli']

import tkinter as tk
from tkinter import simpledialog
import sys
search_url = "sdf"


search_url = f"https://www.google.com/search?hl=en&tbm=isch&q=aibo"


app = Flask(__name__)
@app.route('/')


def get_terminology_summaries():
    """Retrieve and return the first paragraph and a relevant image of each Wikipedia page for each term in the list."""
    technology_terminologies = sorted(technology_terminologiesa)
    result = " \n\n"
    desktop_path = "/Users/ialvector/Desktop/Booklet"
    for current_terminology, term in enumerate(technology_terminologies, start=1):
        wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{term}")
        soup = BeautifulSoup(wikipedia_page.text, 'html.parser')
        paragraphs = soup.find_all('p')
        if paragraphs:
            summary = paragraphs[0].get_text().strip()
            if "Other reasons this message may be displayed:" in summary:
                summary = "No definition found."
        else:
            summary = "No definition found."
        infobox = soup.find('table', {'class': 'infobox'})
        image_tag = None
        if infobox:
            image_tag = infobox.find('img')
        local_image_path = os.path.join(desktop_path, f"{term}.jpeg")
        print(f"Checking local image path: {local_image_path}")

        if os.path.exists(local_image_path):
            image_url = f"http://localhost:8000/{term}.jpeg"
            print(f"Local image found for {term}: {image_url}")
            result += f"<img src='{image_url}' alt='{term}' style='width:200px;height:auto;'><br>"
        else:
            print(f"No local image found for {term}.")
            if image_tag:
                image_url = image_tag['src']
                if image_url.startswith('//'):
                    image_url = "https:" + image_url
                print(f"Image URL for {term}: {image_url}")
                result += f"<img src='{image_url}' alt='{term}' style='width:200px;height:auto;'><br>"
            else:
                print(f"No image found on Wikipedia for {term}.")

        result += f"<br>Terminology Summary {current_terminology} of {len(technology_terminologies)}: {term}<br><br>"
        result += "............................<br>"
        result += f"{summary}<br>"
    return result


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)











for term in technology_terminologies:
    url = search_url + term.replace(" ", "+")  # Replace spaces with '+' for the URL
    webbrowser.open(url)
    time.sleep(20)

technology_terminologies.sort()
for organized_term in technology_terminologies:
    #print(f"'{organized_term}',")
    print("")
mixed_technologies = technology_terminologies.copy()
random.shuffle(mixed_technologies)


# webbrowser.open("https://www.citationmachine.net/apa/cite-a-website")

def is_central_ai_speaking(ai_speak_status):
    global central_ai_speaking
    central_ai_speaking = ai_speak_status


def Centrals_Voicebox(central_response, rate=195):  # You can adjust the default rate here
    print("🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸")
    print("\n\nCentral:", central_response, "\n\n\t-----------------------------------------")
    os.system(f'say -v Boing -r {rate} "{central_response}"')  # Added -r option for rate
    is_central_ai_speaking(True)
    is_central_ai_speaking(False)

# I am at a Job Corps center, and I am a residential student. At about 8pm, residential students can have snacks. I have scanned the barcodes on the most common snacks they provide on the Yuka application, and I am astonished by the number of hazardous additives and their potential effects on human health. One of my pals, who is also residential, got a Rice Krispy, and I informed her that many hazardous additives can cause health disorders, such as potentially carcinogenic and neurotoxic hazardous additives of snacks of hers that I have specifically scanned that are snacks that are commonly being given out. I was hoping you could explain what Yuka is and write an email to the Health and Wellness department to require the kitchen staff to only allow all snacks that are above a nutriscore of 80/100. In the email, also let them know that I will provide below example scans of consumables that the workers of our Job Corps are giving to residential students as common snacks. I would also like it if this could be applied to all foods brought into the cafeteria for students during breakfast, lunch, and dinner since those meals are incredibly influential on the health and cells of the bodies of students. Also have this rule applied to not just snacks and meals but also the hygiene bags provided to new residential students with substances that are applied to their bodies that can also be Endocrine disruptors, allergens, or irritants such as their shampoo, conditioner, body bar soap, toothpaste, mouthwash, shaving cream, tampons, and deodorant. Also not only do the foods for snacks and ingredients for meals need to be scanned before being brought into job corps to feed the students but also the snacks within the vending machines

def get_monitor_info():
    top = 0
    left = 0
    width = 1920
    height = 1080
    return {"top": top, "left": left, "width": width, "height": height}


def remove_alpha_channel(image):
    return image[:, :, :3]


def predict_camera_and_screen_content(frames_per_epoch=30, prediction_delay=2):
    camera_model = MobileNetV2(weights="imagenet")
    screen_model = MobileNetV2(weights="imagenet")
    frame_count = 0

    with mss.mss() as sct:
        monitor_info = get_monitor_info()
        while True:
            _, camera_frame = cv2.VideoCapture(0).read()

            input_shape = (224, 224)
            camera_frame_resized = cv2.resize(camera_frame, input_shape)
            camera_frame_preprocessed = preprocess_input(camera_frame_resized)
            camera_frame_expanded = np.expand_dims(camera_frame_preprocessed, axis=0)
            camera_predictions = camera_model.predict(camera_frame_expanded)
            camera_top_predictions = decode_predictions(camera_predictions, top=3)[0]

            Centrals_Voicebox("Through my camera I see the following:")
            for pred_class, pred_desc, pred_score in camera_top_predictions:
                Centrals_Voicebox(f"{pred_desc}: {pred_score * 100:.2f}%")
            time.sleep(prediction_delay)
            screen_capture = sct.grab(monitor_info)
            screen_frame = np.array(screen_capture)
            screen_frame = screen_frame.astype(np.uint8)
            screen_frame = remove_alpha_channel(screen_frame)
            Image.fromarray(screen_frame).save(f"monitor_{monitor_info['width']}x{monitor_info['height']}.png")
            screen_frame_resized = cv2.resize(screen_frame, input_shape)
            screen_frame_preprocessed = preprocess_input(screen_frame_resized)
            screen_frame_expanded = np.expand_dims(screen_frame_preprocessed, axis=0)
            screen_predictions = screen_model.predict(screen_frame_expanded)
            screen_top_predictions = decode_predictions(screen_predictions, top=3)[0]
            Centrals_Voicebox("Through my screen:")
            for pred_class, pred_desc, pred_score in screen_top_predictions:
                Centrals_Voicebox(f"I see a {pred_desc}: {pred_score * 100:.2f}%")

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    cv2.destroyAllWindows()

#predict_camera_and_screen_content()

def idle_airport_sounds():
    cycle = 0
    frequency = 44100  # Sample rate
    oscillation_duration = 1.0  # Duration of the sound in seconds
    volume = 0.1  # Volume scaling factor (0.0 to 1.0)
    
    while True:
        if not central_ai_speaking:
            formulated_tone = np.linspace(0, oscillation_duration, int(frequency * oscillation_duration), False)
            left_oscillation = 432.0
            right_oscillation = 528.0
            left_waveform = np.sin(2 * np.pi * left_oscillation * formulated_tone) * volume
            right_waveform = np.sin(2 * np.pi * right_oscillation * formulated_tone) * volume
            stereo_waveform = np.column_stack((left_waveform, right_waveform))

            if cycle % 2 == 0:
                sd.play(stereo_waveform, frequency)
                sd.wait()
            cycle += 1
            time.sleep(1.2)
toroid_voicebox = threading.Thread(target=idle_airport_sounds)
toroid_voicebox.start()


def get_wikipedia_summary(term):
    """Retrieve the first paragraph of a Wikipedia page for the given term."""
    wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{term}")
    soup = BeautifulSoup(wikipedia_page.text, 'html.parser')
    paragraphs = soup.find_all('p')
    if paragraphs:
        summary = paragraphs[0].get_text().strip()
        if "Other reasons this message may be displayed:" not in summary:
            return summary
    return "No definition found."





def Wikipedia_Description(Centrals_selected_term):
    wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{Centrals_selected_term}")
    soup_of_desirable_text = BeautifulSoup(wikipedia_page.text, 'html.parser')
    paragraphs = soup_of_desirable_text.find_all('p')
    total_wikipedia_paragraphs = len(paragraphs)
    
    if total_wikipedia_paragraphs >= 3:
      paragraphs = paragraphs[:1]
      total_wikipedia_paragraphs = 3

    elif total_wikipedia_paragraphs <= 2:
        Centrals_Voicebox(f"Term has {total_wikipedia_paragraphs} paragraph: {Centrals_selected_term}")
    summary = "\n".join(paragraph.get_text() + "\n" for paragraph in paragraphs)

    for paragraphs in wikipedia_page:
        try:
            for _ in range(total_terminologies):
                current_terminology = technology_count + 1
                technology = get_next_technology()
            if "Other reasons this message may be displayed:" not in summary:
              #Centrals_Voicebox(f"I now teach you about {technology}")
              print(f"\n\n\t\t     Terminology Summary: {current_terminology} of {total_terminologies}: {technology} \n\t\t\t\t............................ \n{summary}")
        except KeyError:
            continue
        return f"\n\n{soup_of_desirable_text.get_text()}"
    return Centrals_Voicebox("No definition found. Deleting terminology\n")


import wikipedia
wikipedia_random = True
def get_next_technology():
    global technology_count
    if wikipedia_random:
        next_technology = wikipedia.random()
    else:
        next_technology = mixed_technologies[technology_count]
        technology_count = (technology_count + 1) % len(mixed_technologies)
    return next_technology

def get_next_technology():
    global technology_count
    next_technology = mixed_technologies[technology_count]
    technology_count = (technology_count + 1) % len(mixed_technologies)
    return next_technology

total_terminologies = len(mixed_technologies)
total_terminologies = len(technology_terminologies)
probability_of_selecting_one_term = 1 / total_terminologies
probability_as_percentage = round(probability_of_selecting_one_term * 100)

def change_course():
    while True:
        global study_initiative
        study_initiative = input("""Type one of the options:\n________________________\n
  (me)\n (you)\n\t :::""")

        if study_initiative == "you":
            Centrals_Voicebox("I will lead this study today.")
            Centrals_Voicebox(f"All terminologies have an equal {probability_as_percentage:.2f}% probability of being selected")
            break

        elif study_initiative == "me":
            Centrals_Voicebox("Ask me questions for clarification.")
            break

        else:
            Centrals_Voicebox("""You need to type either,
                               (me) if you'd like me to rehearse your terminologies, or
                               (you) if you want me to answer your questions
                               """)
change_course()

def Search_Logic_Only():
    engaging_visual = [
        ' cartoon',
        ' mnemonic device',
        ' gif',
        ' meme',
        ' dark humor',
        ' pun',
        ' cheat sheet',
        ' infographic',
        ' diagram',
        ' 3D model',
        ' tips',
        ' ideas',
        ' '
    ]
    if study_initiative == "you":
        ignored_input = get_next_technology()
        selected_image_type = random.choice(engaging_visual)
        url_image = "https://duckduckgo.com/?q=" + ignored_input + selected_image_type + "&iax=images&ia=images"
        webbrowser.open(url_image)
        Wikipedia_Description(ignored_input)
        # Centrals_Voicebox(f"Any Questions about the terminology {ignored_input}?")

def combine_your_statement_and_generated_text(your_statement, generated_text):
    return your_statement + " " + generated_text

def digital_tone(frequency=50.0, duration=1.0, amplitude=0.1):
    t = np.linspace(0, duration, int(duration * 24100), False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    return waveform

def Rewiring_You():
    global your_statement, previous_generated_text
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 2000
    model_responses = []

    with sr.Microphone() as source:
        sd.play(digital_tone(), 24100)
        print(f"\n\t\t     Central is Listening..\n\t\t🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸\n\n\n\n\n\n\n")


        try:
            #Centrals_Voicebox("One moment...\n\n")
            Search_Logic_Only()
            audio = recognizer.listen(source, timeout=timeout_duration)
            if study_initiative == "me":
                print("You said nothing I'm going to check if you want to type")
                your_statement = input("Your Turn: >>>")

            print("\nYou:", your_statement)

            if your_statement != "":
                api_response = AI_Mindset({"inputs": your_statement})
                model_responses.append(api_response[0]["generated_text"])
                Centrals_potential_transcript = 0
                timed_out = False
                start_time = time.time() 
                while Centrals_potential_transcript < len(model_responses) and not timed_out:
                    api_response = AI_Mindset({"inputs": model_responses[Centrals_potential_transcript]})
                    generated_text = api_response[0]["generated_text"]
                    if generated_text not in model_responses:
                        model_responses.append(generated_text)
                        if generated_text in phrase_counts:
                            phrase_counts[generated_text] += 1
                            if phrase_counts[generated_text] > 3:
                              timed_out = True
                            else:
                              phrase_counts[generated_text] = 1
                        else:
                            phrase_counts[generated_text] = 1
                    Centrals_potential_transcript += 1
                    if time.time() - start_time >= timeout_duration:
                        timed_out = True

                if len(model_responses) == 1:
                    pass

                if timed_out and repeated_phrase_count > 3 and len(phrase_counts) == 1:
                    repeated_phrase_count = max(phrase_counts.values())
                    if repeated_phrase_count > 3 and len(phrase_counts) == 1:
                        Centrals_Voicebox("Hm •• I don't know that one, Timed out")
                else:
                    combined_text = combine_your_statement_and_generated_text(your_statement, model_responses[-1])
                    previous_generated_text = model_responses[-1]
                    Centrals_Voicebox(combined_text.replace(your_statement, ""))

                if combined_text.replace(your_statement, "").strip():
                  your_statement = ""

        except sr.UnknownValueError:
            if study_initiative == "you":
              Centrals_Voicebox("Moving on..")
              Search_Logic_Only()
            pass

        except sr.WaitTimeoutError:
            if not your_statement and study_initiative == "you":
                Centrals_Voicebox("Moving on.....")
                Search_Logic_Only()
            pass

        except ConnectionError:
            Centrals_Voicebox("Connection lost..")
            time.sleep(10)
            pass

        except Exception:
            if your_statement:
              if API_TOKEN == "":
                Centrals_Voicebox("""
                                   Go to https://huggingface.co/settings/tokens and 
                                   get a token. Then copy paste it inside of my API_TOKEN variable above
                                   on code line number 22. Simply paste it in between the empty double 
                                   quotes for me to answer your questions, then restart the terminal.
                                   """)
              else:
                Centrals_Voicebox("ERROR")
            pass

while True:
    Rewiring_You()
