# CENTRAL AI 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸

# Go to https://code.visualstudio.com and install Visual Studio
# Go to https://huggingface.co/settings/tokens and get a token then paste inside API_TOKEN = ""

# Copy paste the following code line in your terminal below:
# pip install SpeechRecognition beautifulsoup4 sounddevice gTTS numpy pyaudio
import speech_recognition as sr
from bs4 import BeautifulSoup
import sounddevice as sd
from gtts import gTTS
import numpy as np
import webbrowser
import threading
import requests
import random
import time
import os

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
API_TOKEN = "" # Insert your token from Hugging Face in between the " "
headers = {"Authorization": f"Bearer {API_TOKEN}"}
study_controller = "you"
central_ai_speaking = False
timeout_duration = 8
user_input = ""
beep_duration = 0.5
technology_counter = 0
technologies_terminologies = [
# Change these terminologies to your liking and according to your field of study
'19 Tauri',
'4th State H2O',
'A/B testing',
'AC motor',
'Acoustic levitation',
'Activation function',
'Adjustable Frequency Drive',
'Agree to disagree',
'Air Drag',
'Alternative hypothesis',
'Area Chart',
'Art of Individualization',
'Associative array',
'Automaton',
'Ball Lightning',
'Bar Chart',
'Biefield-Brown Effect Drive',
'Biefield-brown disk',
'Biefield-brown hemispheres',
'Big O Notation',
'Binary classification',
'Binomial',
'Binomial Distribution',
'Binomial theorem',
'Bootstrapping (statistics)',
'Box Plot',
'Brain-Computer Interface',
'Brain-Computer Interface Communication',
'Brain-Computer Interface Home Use',
'Brain-Machine Interface',
'Brain-to-Brain Interface',
'Brainwave',
'C++',
'CENTRIFUGAL mass',
'Calculus',
'Cathode',
'Cauchy Distribution',
'Cell Nucleus',
'Centrifugal Acceleration',
'Chaos Theory',
'Charge Clusters',
'Chi squared test',
'Chiral Symmetry',
'Chlorophyll',
'Circular polarization',
'Cluster analysis',
'Coefficient',
'Complex Numbers',
'Conductive plate',
'Confidence interval',
'Confusion matrix',
'Conical Coil',
'Conspiracy theory',
'Continuous Distributions',
'Continuous and categorical variables',
'Controversy',
'Convolutional neural networks',
'Cosine similarity',
'Crop Circle Mathematics',
'Data Structures',
'Data and information visualization',
'Data cleansing',
'Data integration',
'Data mining',
'Data munging',
'Data preprocessing',
'Data wrangling',
'Database',
'Degrees Mode',
'Diamagnetic levitation',
'Dielectric',
'Differential Equations',
'Diffusion Limited Aggregation',
'Dimensionality reduction',
'Discrete Distributions',
'Discrete Mathematics',
'Distributive Property',
'Document-term matrix',
'Dr. Eugene Mallove Cold Fusion',
'E8 Crystal',
'Edwin Gray Electric Motor',
'Einstein field equations',
'Electric Field',
'Electrocorticography',
'Electroencephalography',
'Electrogravity',
'Electromagnetic levitation',
'Electronegativity Periodic Table',
'Electronic Centrifuges',
'Electrostatic',
'Energy gradient',
'Ensemble methods',
'Epicyclic gearing',
'Ether technology',
'Evolutionary Programming',
'Exponential Distribution',
'F Distribution',
'F test',
'F-score',
'Feature Matrix',
'Feature engineering',
'Feature scaling',
'Fermentation',
'Ferrite ring magnets',
'Fibonacci Sequence',
'Fleischmann-Pons Cold Fusion',
'Fractal',
'Fractals in Geography',
'Free Energy Device',
'Free Space',
'Frequent pattern mining',
'Gaia, Inc.',
'Gaussian mixture model',
'Generator disks',
'Generator housing',
'Genetic twin',
'Golden Ratio',
'Gradient descent',
'Graduate Record Examinations',
'Graphing calculator',
'Gravifugal Force',
'Gravitational Field',
'Gravitational Flux',
'Gravitational Wave',
'Gravitomagnetism',
'Grid search',
'Ground-Penetrating Radar (GPR)',
'Gut microbiota',
'Hash tables',
'Heatmap',
'Helical coil',
'Hidden layer',
'Hierarchical clustering',
'High Cardinality',
'High Voltage Electrostatics',
'Histogram',
'Holography',
'Homopolar motor',
'Hydrodynamic vortex',
'Hyperspace',
'Hyperspace drive',
'Hypothesis test',
'Independent component analysis',
'Induction motor stator',
'Init method',
'Initialize class',
'Input layer',
'Interpolation',
'Invasive BCI',
'Ion plasma vortex',
'JSON',
'Java (programming language)',
'K-means clustering',
'Ketones',
'L-systems',
'LSTM',
'Laithwaite Engine',
'Law of Sines',
'Lectin',
'Lift factor',
'Light Fidelity',
'Limit (mathematics)',
'Line Chart',
'Linear Equation',
'Linear regression',
'Liquid nitrogen',
'List comprehension',
'Logarithmic Spiral',
'Logistic regression',
'Lognormal Distribution',
'Low-Energy Nuclear Reactions',
'Mach number',
'Magnetic Field',
'Magnetic field stabilizer',
'Magnifying Transmitter',
'Magnus effect',
'Mandelbrot Set',
'Mandelbrots formula',
'Manifold learning',
'Mass flow',
'Mean absolute error',
'Mean squared error',
'Mendocino Motor',
'Microbiome',
'Microscopic Ball Lightning',
'Motor Imagery',
'Multiple linear regression',
'Naive Bayes classifier',
'Natural language processing',
'Navigational radomes',
'Network Theory',
'Neural networks',
'Neurofeedback',
'Newmans energy machine',
'Nikola Tesla',
'Non-invasive BCI',
'Non-parametric methods',
'Nuclear Fusion',
'Nuclear organization',
'Nuclear transfer',
'Null hypothesis',
'One-hot encoding',
'Optical levitation',
'Orion (constellation)',
'Oscillator',
'Outlier detection',
'Output layer',
'Over Unity',
'P300 Speller',
'Parabola',
'Parallel Coordinates',
'Parallel and Concurrent Programming',
'Parent Function',
'Perceptron',
'Performance indicator',
'Permanent Magnets',
'Persistent storage',
'Phyllotaxis',
'Pie Chart',
'Pipeline (computing)',
'Planetary Gear',
'Planetary Rotor',
'Plasma Sheild',
'Platelet Rich Plasma',
'Platonic Solid',
'Pleiades',
'Poisson Distribution',
'Polar Chart',
'Polar Coordinates',
'Polarization state',
'Polarization tensor',
'Poloidal coil',
'Polymath',
'Polynomial',
'Polynomial regression',
'Principal component analysis',
'Probability distribution',
'Propulsion Nacelle',
'Psilocybin',
'Quadratic',
'Quadratic Formula',
'Quasi Crystals',
'R squared',
'ROC curve',
'Radar Chart',
'Radians Mode',
'Random error',
'Random forest classifier',
'Random forest regressor',
'Rapid serial visual presentation',
'Real Number',
'Recurrent neural networks',
'Regenerative Medicine',
'Regularization (mathematics)',
'Reinforcement learning',
'Resampling (statistics)',
'Resonant Frequency',
'Revolutions per minute',
'Ridge regression',
'Rodin coil',
'Root mean squared error',
'Rotating Magnetic Field',
'Rotating Magnetic Field',
'Rotating permanent magnets (alnico)',
'Rotational Fieldgyroscrope',
'Rotational Velocity',
'Sacred Geometry',
'Sankey Diagram',
'Scalar field theory',
'Scalar torodial',
'Scalar–vector–tensor decomposition',
'Scatter Plot',
'Scenar Technologies',
'Searle Disk',
'Self-Similarity',
'Semi-supervised learning',
'Set theory',
'Shanon Entropy',
'Silhouette (clustering)',
'Sine and cosine',
'Slope parameter',
'Softmax function',
'Solenoids',
'Somatic Cell',
'Somatic Cell Nuclear Transfer',
'Stan Meyer Water Fuel Car',
'Standard deviation',
'Starship Coil',
'Statistical classification',
'Statistical inference',
'Statistical tests',
'Stator currents',
'Stem Cells',
'Stereoscopic Rendering',
'Stochastic gradient descent',
'Stubblefield Earth Battery',
'Superconducting cermamic',
'Support vector machine',
'Synaptic pruning',
'Systemic Error',
'T Distribution',
'T test',
'T.T Brown disk',
'TF-IDF',
'Target Vector',
'Target variable',
'Telautomatics',
'Tensor',
'Tensor Holography',
'Tensorflow',
'Tesla Coil',
'Tesla Transformer',
'Tesla Wireless System',
'Teslas Tower',
'Tessellation in nature',
'Testatika',
'Testatika Machine',
'The Terrestrial Stationary Waves',
'The Townsend Brown Electro-Gravity Device',
'Three phase stator windings',
'Time Complexity',
'Time series analysis',
'TlsCAFile',
'Topology',
'Toroidal',
'Torqued Vortex Propulsion System',
'Torsion Physics',
'Torus',
'Touch Sensitive Paint',
'Training, validation, and test data sets',
'Transfer learning',
'Tri-arcuate ballistic electrode',
'Unified Field Quadrangle',
'Unit Circle',
'Univariate analysis',
'Vacuum Triode',
'Variance',
'Vortex wave generator',
'Waterfall Chart',
'Wave Phenomena',
'Word Cloud',
'Word embedding',
'XGBoost',
'Yttrium barium-copper oxide',
'Z score',
'Z test',
'Zero Point Energy',
'Zeta Reticuli',
'Zoeppritz equations'
]
technologies_terminologies.sort()
print("Organized terms:")
for organized_term in technologies_terminologies:
    print(f"'{organized_term}',")
mixed_technologies = technologies_terminologies.copy()
random.shuffle(mixed_technologies)


def set_central_ai_speaking(ai_speak_status):
    global central_ai_speaking
    central_ai_speaking = ai_speak_status


def CentralAI_Voicebox(central_response):
    voicebox = gTTS(text=central_response, lang="en")
    print("🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸")
    print("\n\nCentral AI:", central_response)
    word_count = len(central_response.split())
    print("Word Count:", word_count)
    print("\n\n")
    haptic_file_name = "digital_voice_output.mp3"
    voicebox.save(haptic_file_name)
    set_central_ai_speaking(True)
    os.system(f"afplay {haptic_file_name}")
    set_central_ai_speaking(False)


def idle_beeps():
    run = 0
    while True:
        if not central_ai_speaking:
            sample_rate = 44100
            t = np.linspace(0, beep_duration, int(sample_rate * beep_duration), False)
            frequency = 440.0
            torus_radius = 0.2
            torus_center = 0.5
            torus_waveform = np.sin(2 * np.pi * frequency * t)
            torus_waveform *= np.sin(2 * np.pi * torus_radius * np.cos(2 * np.pi * torus_center * t))
            torus_waveform *= 0.3
            if run % 2 == 0:
                sd.play(torus_waveform, sample_rate)
                sd.wait()
            run += 1
            time.sleep(1.2)
beep_thread = threading.Thread(target=idle_beeps)
beep_thread.start()


def read_wikipedia_open_visuals(AI_selected_data_term):
    wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{AI_selected_data_term}")
    soup = BeautifulSoup(wikipedia_page.text, 'html.parser')
    paragraphs = soup.find_all('p')
    max_paragraphs = len(paragraphs)
    paragraphs = paragraphs[:max_paragraphs]
    summary = "\n".join(paragraph.get_text() + "\n" for paragraph in paragraphs)
    for paragraphs in wikipedia_page:
        try:
            for _ in range(total_technologies):
                terminology_number = technology_counter + 1
                technology = get_next_technology()
            if "Other reasons this message may be displayed:" not in summary:
              CentralAI_Voicebox(f"Jot down questions on a notepad for me to answer, after I teach you about {technology}")
              CentralAI_Voicebox(f"\n\n\t\t     Wikipedia Summary: Terminology {terminology_number} of {total_technologies}: {technology},,, 📖 \n\t\t\t\t............................\n {summary}")
            print("Summary:", summary)
            print(max_paragraphs)
        except KeyError:
            continue
        return f"\n\n{soup.get_text()}"
    return CentralAI_Voicebox("No definition found. Deleting terminology\n")


def get_next_technology():
    global technology_counter
    next_technology = mixed_technologies[technology_counter]
    technology_counter = (technology_counter + 1) % len(mixed_technologies)
    return next_technology
total_technologies = len(mixed_technologies)


CentralAI_Voicebox("Who is in control of our study session, me or you")
print("Type (me) or (you)")
study_controller = input()
if study_controller == "you":
    CentralAI_Voicebox("Okay! I will lead this study session")
if study_controller == "me":
    CentralAI_Voicebox("Okay! Take control. Let us study together for this session.")


def Bio_Logic_silence():
    engaging_visual = [
        ' cartoon',
        ' mnemonic device',
        ' joke',
        ' visualization',
        ' gif',
        ' meme',
        ' dark humor',
        ' comic',
        ' cheat sheet'
    ]
    if study_controller == "you":
        ignored_input = get_next_technology()
        selected_image_type = random.choice(engaging_visual)
        url_image = "https://duckduckgo.com/?q=" + ignored_input + selected_image_type + "&iax=images&ia=images"
        AI_interviewer = "https://jobinterview.coach/app-home"
        print("Enhance your employability:", AI_interviewer)
        webbrowser.open(f"https://en.wikipedia.org/wiki/{ignored_input}")
        webbrowser.open(url_image)
        read_wikipedia_open_visuals(ignored_input)
        CentralAI_Voicebox(f"Any Questions about the terminology {ignored_input}?")
    else:
        print("Please type either 'me' or 'you'.")


def AI_Mindset(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    json_data = response.json()
    while "next" in response.links.keys():
        json_data.extend(response.json())
        print(json_data.extend(response.json()))
    return json_data


def combine_user_input_and_generated_text(user_input, generated_text):
    return user_input + " " + generated_text


previous_generated_text = ""


def Mind_Scan_Bio_Logic():
    global user_input, previous_generated_text
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    api_responses = []
    with sr.Microphone() as source:
        print("\n\t\t     Central AI is Listening..\n\t\t🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸\n\n\n\n\n\n\n\n\n\n\n\n")
        try:
            audio = recognizer.listen(source, timeout=timeout_duration)
            if audio:
                user_input = recognizer.recognize_google(audio)
            print(f"You: {user_input}")
            if user_input != "":
                api_response = AI_Mindset({"inputs": user_input})
                api_responses.append(api_response[0]["generated_text"])
                i = 0
                while i < len(api_responses):
                    api_response = AI_Mindset({"inputs": api_responses[i]})
                    generated_text = api_response[0]["generated_text"]
                    if generated_text not in api_responses:
                        api_responses.append(generated_text)
                    i += 1
                if len(api_responses) == 1:
                    pass
            
                combined_text = combine_user_input_and_generated_text(user_input, api_responses[-1])
                CentralAI_Voicebox(combined_text.replace(user_input, ""))
                previous_generated_text = api_responses[-1]
                if combined_text.replace(user_input, "").strip():
                  user_input = ""

        except sr.UnknownValueError:
            if study_controller == "you":
              CentralAI_Voicebox("Moving on")
              Bio_Logic_silence()
            pass

        except sr.WaitTimeoutError:
            if not user_input and study_controller == "you":
                CentralAI_Voicebox("Moving on")
                Bio_Logic_silence()
            pass

        except Exception:
            if user_input:
              if API_TOKEN == "":
                CentralAI_Voicebox("""
                                   You need to go to https://huggingface.co/settings/tokens and 
                                   get a token. You then must copy paste it inside of my API TOKEN above
                                   on code line number 21. Simply paste it in between the empty double 
                                   quotes for me to answer your questions then restart the terminal.
                                   """)
              else:
                CentralAI_Voicebox("I am unable to answer your question because I am disconnected..")
            pass
        

while True:
    Mind_Scan_Bio_Logic()
