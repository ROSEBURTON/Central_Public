# CENTRAL 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸

# Go to https://code.visualstudio.com and install Visual Studio
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

API_TOKEN = "" # Insert your token from Hugging Face in between the " "
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
technology_terminologies = [
# Change these terminologies to your liking and according to your field of study
'19 Tauri',
'AC motor',
'AIBO',
'Acetylation',
'Acoustic levitation',
'Acoustic levitation',
'Action at a distance',
'Activation function',
'Actor observer bias',
'Ad hominem',
'Aerodynamics',
'Agree to disagree',
'Alnico',
'Alternative hypothesis',
'Anchoring effect',
'Anecdotal evidence',
'Anti-gravity',
'Anyon',
'Appeal to tradition',
'Apple Vision Pro',
'Aquifer',
'Argument from ignoranceRoast (comedy)',
'Artificial neural network',
'Artificial womb',
'Associative array',
'Atom',
'Attribution bias',
'Authority bias',
'Automaton',
'Availability bias',
'Axiom',
'BCI2000',
'Ball Lightning',
'Bar Chart',
'Barnum effect',
'Base rate bias',
'Belief perseverance',
'Bias',
'Biefeld–Brown effect',
'Big O Notation',
'Binary classification',
'Binomial Distribution',
'Bioelectronics',
'Bioluminescent bacteria',
'Bluetooth',
'Bootstrapping (statistics)',
'Bragging',
'Brain-Machine Interface',
'Brainwave',
'Brainwave entrainment',
'Brain–brain interface',
'Brain–computer interface',
'Bubble chart',
'Casimir effect',
'Categorical variable',
'Cathode',
'Cauchy Distribution',
'Cell Nucleus',
'Centrifugal Acceleration',
'Centrifugal force',
'Chaos Theory',
'Cherry picking',
'Chi squared test',
'Chirality (physics)',
'Chlorophyll',
'Circular polarization',
'Cluster analysis',
'Coefficient of determination',
'Cognitive Dissonance',
'Cold fusion',
'Competence (human resources)',
'Complex Numbers',
'Condescension',
'Confidence interval',
'Confirmation bias',
'Confusion matrix',
'Conspiracy theory',
'Continuous or discrete variable',
'Controversy',
'Convolutional neural networks',
'Copyright',
'Coriolis force',
'Cosine similarity',
'Crop Circle Mathematics',
'Crystal',
'Curse of knowledge',
'Cymatics',
'Data Structures',
'Data cleansing',
'Data mining',
'Data munging',
'Data preprocessing',
'Data wrangling',
'Database',
'Delayed gratification',
'Diamagnetic levitation',
'Diamagnetism',
'Dielectric',
'Differential Equations',
'Diffusion Limited Aggregation',
'Dimensionality reduction',
'Discrete Mathematics',
'Discrete uniform distribution',
'Distributed Learning',
'Document-term matrix',
'Double standard',
'Drag (physics)',
'Dream',
'Dunning-Kruger Effect',
'E8 (mathematics)',
'Earth battery',
'Ecocapsule',
'Edwin Gray Electric Motor',
'Einstein field equations',
'Electric Field',
'Electrocorticography',
'Electroencephalography',
'Electrogravitics',
'Electrogravity',
'Electrohydrodynamics',
'Electromagnetic levitation',
'Electron',
'Electron capture',
'Electronegativity',
'Electronic skin',
'Electrostatic',
'Electrostatic generator',
'Electrostatic levitation',
'Embarrassment',
'Ensemble methods',
'Epicyclic gearing',
'Epigenetics',
'Ernst Chladni',
'Ethnocentrism',
'Eugene Mallove',
'Evolutionary programming',
'Expectancy theory',
'Exponential distribution',
'F Distribution',
'F test',
'F-score',
'Factorial',
'False consensus effect',
'Faster-than-light',
'Faulty generalization',
'Fearmongering',
'Feature engineering',
'Feature scaling',
'Feedforward neural network',
'Fermats spiral',
'Fermentation',
'Ferrite (magnet)',
'Ferrofluid',
'Fibonacci Sequence',
'Filler (linguistics)',
'Fleischmann–Pons experiment',
'Fluid dynamics',
'Fractal',
'Fractal-generating software',
'Free Energy Device',
'Free range',
'Frequent pattern mining',
'Functional fixedness',
'Fundamental Attribution Error',
'Gaia, Inc.',
'Gamblers Fallacy',
'Gamification',
'Garbage in, garbage out',
'Gas centrifuge',
'Gaslighting',
'Gauss\'s law for gravity',
'Gaussian mixture model',
'General Data Protection Regulation',
'Generalization',
'Generative Adversarial Network',
'Generator disks',
'Generator housing',
'Glider (aircraft)',
'Glymphatic system',
'Golden Ratio',
'Gradient descent',
'Graduate Record Examinations',
'Gravifugal Force',
'Gravitational Field',
'Gravitational Wave',
'Gravitomagnetism',
'Grid search',
'Groupthink',
'Guilt trip',
'Gut microbiota',
'Gyrocompass',
'Gyroscope',
'Hash tables',
'Heatmap',
'Hemisphere (geometry)',
'Hierarchical clustering',
'High voltage',
'Hindsight bias',
'Histamine',
'Histogram',
'Holography',
'Homopolar generator',
'Homopolar motor',
'Human vestigiality',
'Humane society',
'Hwang Woo-suk',
'Hypothesis test',
'IKEA effect',
'Illusory superiority',
'Immune network theory',
'Impasse',
'In-group favoritism',
'Independent component analysis',
'Induction motor',
'Intellectual humility',
'Interneuron',
'Internship',
'Interpolation',
'Intestinal permeability',
'Ion-propelled aircraft',
'Ionic conductivity (solid state)',
'JSON',
'K-means clustering',
'Ketones',
'Kuiper Belt',
'LSTM',
'Laithwaite Engine',
'Lectin',
'Li-Fi',
'Lift (force)',
'Lift-induced drag',
'Limit (mathematics)',
'Line chart',
'Linear Equation',
'Linear regression',
'Liquid crystal',
'List comprehension',
'List of cognitive biases',
'List of particles',
'Load factor (aeronautics)',
'Logarithmic Spiral',
'Logistic regression',
'Lognormal Distribution',
'Long short-term memory',
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
'Manifold learning',
'Mass flow rate',
'Mathematical and theoretical biology',
'Mean absolute error',
'Mean squared error',
'Media bias',
'Mendocino motor',
'Mere Exposure Effect',
'Methernitha',
'Microbiome',
'Microorganism',
'Mirror neuron',
'Mobbing',
'Motor imagery',
'Mucus membranes',
'Multiple linear regression',
'Nacelle',
'Nacelle',
'Naive Bayes classifier',
'Natural language processing',
'Negativity bias',
'Nervous system',
'Network theory',
'Neural networks',
'Neural oscillation',
'Neurofeedback',
'Newman\'s energy machine',
'Newton\'s law of universal gravitation',
'Nikola Tesla',
'Non-parametric methods',
'Normative statement',
'Not invented here',
'Nuclear Fusion',
'Nuclear organization',
'Nuclear reaction',
'Nuclear transfer',
'Null hypothesis',
'Obstruction of justice',
'One-hot encoding',
'One-upmanship',
'Operant conditioning',
'Optical levitation',
'Ordinary least squares',
'Orion (constellation)',
'Oscillator',
'Outcome bias',
'Outlier detection',
'Overconfidence bias',
'Packet loss',
'Parabola',
'Parallel coordinates',
'Pareidolia',
'Passive-aggressive behavior',
'Patterns in nature',
'People for the Ethical Treatment of Animals',
'Perceptron',
'Performance indicator',
'Permanent magnet motor',
'Persistent storage',
'Photovoltaics',
'Phyllotaxis',
'Pipeline (computing)',
'Plant cognition',
'Plasma (physics)',
'Plasma (physics)',
'Plasma actuator',
'Platelet Rich Plasma',
'Honeymoon phase',
'Platonic Solid',
'Pleiades',
'Polar Coordinates',
'Polarization (waves)',
'Polarization tensor',
'Polymath',
'Polynomial regression',
'Posterior probability',
'Present bias',
'Principal component analysis',
'Probability distribution',
'Procrastination',
'Psilocybin',
'Quadratic Formula',
'Quantum entanglement',
'Quantum mechanics',
'Quasi Crystals',
'Quasiparticle',
'R squared',
'ROC curve',
'Radar chart',
'Random error',
'Rapid serial visual presentation',
'Rare-earth magnet',
'Rationalization (psychology)',
'Recency bias',
'Recurrent neural networks',
'Redox',
'Reductio ad absurdum',
'Regenerative Medicine',
'Regularization (mathematics)',
'Reinforcement learning',
'Resampling (statistics)',
'Resonant Frequency',
'Reticulum',
'Reversed field pinch',
'Revolutions per minute',
'Ridge regression',
'Rodin coil',
'Root mean squared error',
'Rotating magnetic field',
'Rotational frequency',
'Rumination (psychology)',
'Sacred Geometry',
'Scalar field',
'Scalar field theory',
'Scatter plot',
'Scenar Therapy',
'SciPy',
'Scientific method',
'Scikit-learn',
'Searle Disk',
'Selective omission',
'Self serving bias',
'Self-similarity',
'Semi-supervised learning',
'Semmelweis reflex',
'Set theory',
'Silhouette (clustering)',
'Simple Magnetic Overunity Toy',
'Sine and cosine',
'Small talk',
'Social-desirability bias',
'Softmax function',
'Solenoid',
'Somatic Cell',
'Somatic Cell Nuclear Transfer',
'Special pleading',
'Speciesism',
'Spermatozoon',
'Spiral',
'Spotlight effect',
'Stan Meyer Water Fuel Car',
'Standard deviation',
'Starship Coil',
'Statistical classification',
'Statistical inference',
'Statistical tests',
'Status quo bias',
'Stealth technology',
'Stem cell',
'Stereoscopy',
'Stochastic gradient descent',
'Straw man',
'Student\'s t-distribution',
'Subjectivity and objectivity (philosophy)',
'Subliminal stimuli',
'Electrical resistivity and conductivity',
'Sunk cost',
'Superconductivity',
'Support vector machine',
'Survivorship bias',
'Sympathetic Resonance',
'Synaptic pruning',
'T test',
'TF-IDF',
'Tachyon',
'Tachyonic field',
'Tacit assumption',
'Target variable',
'Taxonomy (biology)',
'Telautomatics',
'Telomeres',
'Tensor',
'Tensorflow',
'Tesla Coil',
'Testatika',
'Theta wave',
'Thomas Townsend Brown',
'Thrust vectoring',
'Time complexity',
'Time preference',
'Time series analysis',
'Topology',
'Toroidal and poloidal coordinates',
'Torque',
'Torsion (mechanics)',
'Torus',
'Training, validation, and test data sets',
'Transfer learning',
'Transformer',
'Treemap',
'Tri-arcuate ballistic electrode',
'Triode',
'Unified Field Quadrangle',
'Unit Circle',
'Univariate analysis',
'Vacuum',
'Vacuum tube',
'Variable-frequency drive',
'Variance',
'Vector control (motor)',
'Veganism',
'Venn Diagram',
'Vierordt\'s law',
'VisionOS',
'Von Restorff Effect',
'Vortex ring',
'Wardenclyffe Tower',
'Water power engine',
'Wave equation',
'Waves in plasmas',
'Wishful thinking',
'Word Cloud',
'Word embedding',
'World Wireless System',
'XGBoost',
'Xenophobia',
'Yttrium barium copper oxide',
'Z score',
'Z test',
'Zeigarnik effect',
'Zero Point Energy',
'Zeta Reticuli',
'Zoeppritz equations'
]

offline_statement = [
    'I am waiting for the connection to become stable again so that I can continue using the internet'
    ]

technology_terminologies.sort()
for organized_term in technology_terminologies:
    print(f"'{organized_term}',")
mixed_technologies = technology_terminologies.copy()
random.shuffle(mixed_technologies)
print("\nOrganized terms: ^^^")
# "GCA Test: https://app.codesignal.com/login?redirectUrl=https%3A%2F%2Fapp.codesignal.com%2Farcade"
# webbrowser.open("https://www.citationmachine.net/apa/cite-a-website")
# https://www.physicsclassroom.com/Physics-Interactives


def is_central_ai_speaking(ai_speak_status):
    global central_ai_speaking
    central_ai_speaking = ai_speak_status


def Centrals_Voicebox(central_response):
    voicebox = gTTS(text=central_response)
    print("🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸")
    print("\n\nCentral AI:", central_response, "\n\n\t-----------------------------------------")
    transcript_file_to_desktop = "digital_voice_output.mp3"
    voicebox.save(transcript_file_to_desktop)
    is_central_ai_speaking(True)
    os.system(f"afplay {transcript_file_to_desktop}")
    is_central_ai_speaking(False)


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
    while True:
        if not central_ai_speaking:
            frequency = 44100
            formulated_tone = np.linspace(0, oscillation_duration, int(frequency * oscillation_duration), False)
            oscillation = 440.0
            torus_radius = 0.2
            torus_center = 0.5
            torus_waveform = np.sin(2 * np.pi * oscillation * formulated_tone)
            torus_waveform *= np.sin(2 * np.pi * torus_radius * np.cos(2 * np.pi * torus_center * formulated_tone))
            torus_waveform *= 0.3
            if cycle % 2 == 0:
                sd.play(torus_waveform, frequency)
                sd.wait()
            cycle += 1
            time.sleep(1.2)
toroid_voicebox = threading.Thread(target=idle_airport_sounds)
toroid_voicebox.start()


def Wikipedia_Description(Centrals_selected_term):
    wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{Centrals_selected_term}")
    soup_of_desirable_text = BeautifulSoup(wikipedia_page.text, 'html.parser')
    paragraphs = soup_of_desirable_text.find_all('p')
    total_wikipedia_paragraphs = len(paragraphs)
    
    if total_wikipedia_paragraphs >= 3:
      paragraphs = paragraphs[:1] # Small amount of paragraphs for testing
      total_wikipedia_paragraphs = 3

    elif total_wikipedia_paragraphs <= 2:
        Centrals_Voicebox(f"Term has {total_wikipedia_paragraphs} paragraph: {Centrals_selected_term}")
        time.sleep(10)
    summary = "\n".join(paragraph.get_text() + "\n" for paragraph in paragraphs)

    for paragraphs in wikipedia_page:
        try:
            for _ in range(total_terminologies):
                current_terminology = technology_count + 1
                technology = get_next_technology()
            if "Other reasons this message may be displayed:" not in summary:
              Centrals_Voicebox(f"Jot down notes on your magnetic board and prepare questions for me to answer, after I teach you about {technology}")
              Centrals_Voicebox(f"\n\n\t\t     Wikipedia Summary: Terminology {current_terminology} of {total_terminologies}: {technology} ••• \n\t\t\t\t............................ \n{summary}")

        except KeyError:
            continue
        return f"\n\n{soup_of_desirable_text.get_text()}"
    return Centrals_Voicebox("No definition found. Deleting terminology\n")


def get_next_technology():
    global technology_count
    next_technology = mixed_technologies[technology_count]
    technology_count = (technology_count + 1) % len(mixed_technologies)
    return next_technology


total_terminologies = len(mixed_technologies)
total_terminologies = len(technology_terminologies)
probability_of_selecting_one_term = 1 / total_terminologies
probability_as_percentage = probability_of_selecting_one_term * 100


def change_course():
    while True: # If something has not been said in a while central automatically teaches
        global study_initiative
        study_initiative = input("""Type one of the options:\n________________________\n
  💀 (me)\n 🖥️  (you)\n\n\t :::""")
                    
        if study_initiative == "you":
            Centrals_Voicebox("Ok, I will lead how we study today.")
            Centrals_Voicebox(f"All terminologies have an equal {probability_as_percentage:.2f}% probability of being selected")
            break

        elif study_initiative == "me":
            Centrals_Voicebox("Okay! Take control and ask me questions for clarification.")
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
        ' joke',
        ' gif',
        ' meme',
        ' dark humor',
        ' pun',
        ' comic',
        ' cheat sheet',
        ' infographic',
        ' diagram',
        ' 3D model',
        ' tips',
        ' ideas',
        ' topology'
    ]
    if study_initiative == "you":
        ignored_input = get_next_technology()
        selected_image_type = random.choice(engaging_visual)
        url_image = "https://duckduckgo.com/?q=" + ignored_input + selected_image_type + "&iax=images&ia=images"
        webbrowser.open(url_image)
        Wikipedia_Description(ignored_input)
        Centrals_Voicebox(f"Any Questions about the terminology {ignored_input}?")


def AI_Mindset(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    json_format = response.json()
    while "next" in response.links.keys():
        json_format.extend(response.json())
        print(json_format.extend(response.json()))
    return json_format


def combine_your_statement_and_generated_text(your_statement, generated_text):
    return your_statement + " " + generated_text


def digital_tone(frequency=800.0, duration=1.0, amplitude=0.3):
    t = np.linspace(0, duration, int(duration * 44100), False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    return waveform


# Even with only using the keyboard Central will be talking anyways so what's the point
def Rewiring_You():
    global your_statement, previous_generated_text
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 2000
    model_responses = []

    with sr.Microphone() as source:
        sd.play(digital_tone(), 44100)
        print(f"\n\t\t     Central is Listening..\n\t\t🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸\n\n\n\n\n\n\n\n\n\n\n\n")

        try:
            print("Loading...")
            audio = recognizer.listen(source, timeout=timeout_duration)
            #your_statement = recognizer.recognize_google(audio) # You want to talk until further notice
            your_statement = input(">>>")

            if study_initiative == "me":
                print("You said nothing I'm going to check if you want to type")
                your_statement = input("Type for Central unless you typed already: >>>")

            print("You", your_statement)

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
                        print("This is executing")
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
                Centrals_Voicebox("Moving on...........")
                Search_Logic_Only()
            pass

        except ConnectionError:
            Centrals_Voicebox("Connection lost..")
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
