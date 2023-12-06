# CENTRAL AI 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸

# Go to https://code.visualstudio.com and install Visual Studio
# Go to https://huggingface.co/settings/tokens and get a token then paste inside API_TOKEN = ""
# Copy paste the following code line in your terminal below:
# pip install SpeechRecognition beautifulsoup4 sounddevice gTTS numpy pyaudio

from requests.exceptions import ConnectionError
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
API_SECRET = "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" # Insert your token from Hugging Face in between the " "
headers = {"Authorization": f"Bearer {API_SECRET}"}
# study_initiative = "you"
central_ai_speaking = False
voicebox_timeout = 20
timeout_duration = 9
phrase_counts = {}
user_input = ""
oscillation_duration = 0.5
technology_count = 0
technology_terminologies = [
# Change these terminologies to your liking and according to your field of study
'19 Tauri',
'4th State H2O',
'A/B testing',
'AC motor',
'AIBO',
'Acetylation',
'Acoustic levitation',
'Action at a distance',
'Activation function',
'Actor observer bias',
'Ad hominem',
'Adjustable Frequency Drive',
'Agree to disagree',
'Air Drag',
'Alternative hypothesis',
'Anecdotal evidence',
'Apple Vision Pro',
'Area chart',
'Art of Individualization',
'Artificial neural network',
'Associative array',
'Attribution bias',
'Authority Bias',
'Automaton',
'Availability bias',
'Backfire bias',
'Ball Lightning',
'Bar Chart',
'Barnum Effect',
'Base rate bias',
'Better than average bias',
'Biefield-Brown Effect Drive',
'Biefield-brown disk',
'Biefield-brown hemispheres',
'Big O Notation',
'Binary bias',
'Binary classification',
'Binomial Distribution',
'Binomial theorem',
'Bioluminescent bacteria',
'Bluetooth',
'Bootstrapping (statistics)',
'Box Plot',
'Bragging',
'Brain-Computer Interface Communication',
'Brain-Computer Interface Home Use',
'Brain-Machine Interface',
'Brain-to-Brain Interface',
'Brainwave',
'Brain–computer interface',
'Bubble Chart',
'Bullet Graph',
'C++',
'Cathode',
'Cauchy Distribution',
'Cell Nucleus',
'Centrifugal Acceleration',
'Centrifugal force',
'Chaos Theory',
'Charge Clusters',
'Chi squared test',
'Chiral Symmetry',
'Chlorophyll',
'Circular polarization',
'Cluster analysis',
'Coefficient',
'Cognitive Dissonance',
'Colombo method',
'Complex Numbers',
'Conductive plate',
'Confidence interval',
'Confirmation bias',
'Confusion matrix',
'Conical Coil',
'Conspiracy theory',
'Continuous and categorical variables',
'Controversy',
'Convolutional neural networks',
'Cosine similarity',
'Crop Circle Mathematics',
'Curse of knowledge',
'Cymatics',
'Data Structures',
'Data cleansing',
'Data integration',
'Data mining',
'Data munging',
'Data preprocessing',
'Data wrangling',
'Database',
'Degrees Mode',
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
'Distributive Property',
'Document-term matrix',
'Double standard',
'Doughnut Chart',
'Dr. Eugene Mallove Cold Fusion',
'Dunning-Kruger Effect',
'E8 Crystal',
'Ecocapsule',
'Edwin Gray Electric Motor',
'Einstein field equations',
'Electric Field',
'Electrocorticography',
'Electroencephalography',
'Electrogravitics',
'Electrogravity',
'Electromagnetic levitation',
'Electron',
'Electronegativity Periodic Table',
'Electronic Centrifuges',
'Electronic skin',
'Electrostatic',
'Electrostatic levitation',
'Energy gradient',
'Ensemble methods',
'Epicyclic gearing',
'Ernst Chladni',
'Ether technology',
'Evolutionary Programming',
'Expectancy theory',
'Exponential Distribution',
'F Distribution',
'F test',
'F-score',
'False Consensus Effect',
'Faster-than-light',
'Faulty generalization',
'Fearmongering',
'Feature Matrix',
'Feature engineering',
'Feature scaling',
'Fermats spiral',
'Fermentation',
'Ferrite ring magnets',
'Fibonacci Sequence',
'Fleischmann-Pons Cold Fusion',
'Fractal',
'Fractals in Geography',
'Free Energy Device',
'Frequent pattern mining',
'Functional Fixedness',
'Fundamental Attribution Error',
'Gaia, Inc.',
'Gamblers Fallacy',
'Gamification',
'Gaslighting',
'Gaussian mixture model',
'Gausss law for gravity',
'Generalization',
'Generator disks',
'Generator housing',
'Genetic twin',
'Glymphatic system',
'Golden Ratio',
'Gradient descent',
'Graduate Record Examinations',
'Graphing calculator',
'Gravifugal Force',
'Gravitational Field',
'Gravitational Wave',
'Gravitomagnetism',
'Grid search',
'Ground-Penetrating Radar (GPR)',
'Groupthink',
'Gut microbiota',
'Hash tables',
'Heatmap',
'Helical coil',
'Hidden layer',
'Hierarchical clustering',
'High Cardinality',
'High Voltage Electrostatics',
'Hindsight bias',
'Histamine',
'Histogram',
'Holography',
'Homopolar motor',
'Human multitasking',
'Hwang Woo-suk',
'Hydrodynamic vortex',
'Hypothesis test',
'IKEA Effect',
'Impasse',
'Independent component analysis',
'Induction motor stator',
'Init method',
'Instant Gratification',
'Intellectual humility',
'Internship',
'Interpolation',
'Intestinal permeability',
'Ion plasma vortex',
'JSON',
'K-means clustering',
'Ketones',
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
'Manifold learning',
'Mass flow',
'Mean absolute error',
'Mean squared error',
'Mendocino Motor',
'Mere Exposure Effect',
'Microbiome',
'Mirror neuron',
'Motor Imagery',
'Mucus membranes',
'Multiple linear regression',
'Naive Bayes classifier',
'Natural language processing',
'Navigational radomes',
'Negativity bias',
'Nervous system',
'Network Theory',
'Neural networks',
'Neural oscillation',
'Neurofeedback',
'Newmans energy machine',
'Newtons law of universal gravitation',
'Nikola Tesla',
'Non-invasive BCI',
'Non-parametric methods',
'Normative statement',
'Not-Invented-Here (NIH) Bias',
'Nuclear Fusion',
'Nuclear organization',
'Nuclear transfer',
'Null hypothesis',
'One-hot encoding',
'Optical levitation',
'Ordinary least squares',
'Orion (constellation)',
'Oscillator',
'Outcome bias',
'Outlier detection',
'Over Unity',
'Overconfidence bias',
'P300 Speller',
'Pair Plot',
'Parabola',
'Parallel Coordinates',
'Parallel and Concurrent Programming',
'Pareidolia',
'Parent Function',
'Patterns in nature',
'Perceptron',
'Performance indicator',
'Permanent Magnets',
'Persistent storage',
'Photovoltaics',
'Phyllotaxis',
'Pipeline (computing)',
'Planetary Gear',
'Planetary Rotor',
'Plant cognition',
'Plasma (physics)',
'Plasma Sheild',
'Platelet Rich Plasma',
'Platonic Solid',
'Pleiades',
'Poisson Distribution',
'Polar Coordinates',
'Polarization state',
'Polarization tensor',
'Poloidal coil',
'Polymath',
'Polynomial',
'Polynomial regression',
'Positive statement',
'Present bias',
'Principal component analysis',
'Probability distribution',
'Procrastination',
'Propulsion Nacelle',
'Pseudoscience',
'Psilocybin',
'Quadratic',
'Quadratic Formula',
'Quasi Crystals',
'Quasi Particle',
'R squared',
'ROC curve',
'Radar Chart',
'Radio frequency',
'Random error',
'Rapid serial visual presentation',
'Rationalization (psychology)',
'Real Number',
'Recency bias',
'Recurrent neural networks',
'Regenerative Medicine',
'Regularization (mathematics)',
'Reinforcement learning',
'Resampling (statistics)',
'Resonant Frequency',
'Reticulum',
'Revolutions per minute',
'Ridge regression',
'Rodin coil',
'Root mean squared error',
'Rotating Magnetic Field',
'Rotating Magnetic Field',
'Rotating permanent magnets (alnico)',
'Rotational Fieldgyroscrope',
'Rotational Velocity',
'Rumination',
'Sacred Geometry',
'Sankey Diagram',
'Scalar field theory',
'Scalar torodial',
'Scatter Plot',
'Scenar Technologies',
'Scientific method',
'Searle Disk',
'Selective omission',
'Self efficacing bias',
'Self serving bias',
'Self-Similarity',
'Semi-supervised learning',
'Semmelweis Reflex',
'Set theory',
'Shanon Entropy',
'Silhouette (clustering)',
'Sine and cosine',
'Slope parameter',
'Small talk',
'Social Desirability Bias',
'Softmax function',
'Solenoids',
'Somatic Cell',
'Somatic Cell Nuclear Transfer',
'Speciesism',
'Spermatozoon',
'Spotlight Effect',
'Stan Meyer Water Fuel Car',
'Standard deviation',
'Starship Coil',
'Statistical classification',
'Statistical inference',
'Statistical tests',
'Stator currents',
'Status quo Bias',
'Stealth technology',
'Stem Cells',
'Stem cell',
'Stem cell',
'Stereoscopic Rendering',
'Stochastic gradient descent',
'Straw man',
'Stubblefield Earth Battery',
'Subjective',
'Subjective',
'Sunk cost bias',
'Superconducting cermamic',
'Support vector machine',
'Survivorship Bias',
'Sympathetic Resonance',
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
'Teslas Tower',
'Testatika',
'Testatika Machine',
'The Terrestrial Stationary Waves',
'The Townsend Brown Electro-Gravity Device',
'Theta wave',
'Three phase stator windings',
'Time Complexity',
'Time Discounting',
'Time series analysis',
'TlsCAFile',
'Topology',
'Toroidal',
'Torqued Vortex Propulsion System',
'Torsion Physics',
'Torus',
'Training, validation, and test data sets',
'Transfer learning',
'Treemap',
'Tri-arcuate ballistic electrode',
'Unified Field Quadrangle',
'Unit Circle',
'Univariate analysis',
'Vacuum',
'Vacuum Triode',
'Variance',
'Venn Diagram',
'Vierordts law',
'VisionOS',
'Von Restorff Effect',
'Vortex wave generator',
'Waterfall Chart',
'Wave Phenomena',
'Word Cloud',
'Word embedding',
'World Wireless System',
'XGBoost',
'Yttrium barium-copper oxide',
'Z score',
'Z test',
'Zeigarnik Effect',
'Zero Point Energy',
'Zeta Reticuli',
'Zoeppritz equations'
]

technology_terminologies.sort()
print("Organized terms:")
for organized_term in technology_terminologies:
    print(f"'{organized_term}',")
mixed_technologies = technology_terminologies.copy()
random.shuffle(mixed_technologies)

# webbrowser.open("https://www.citationmachine.net/apa/cite-a-website")

def is_central_ai_speaking(ai_speak_status):
    global central_ai_speaking
    central_ai_speaking = ai_speak_status


def Centrals_Voicebox(central_response):
    voicebox = gTTS(text=central_response)
    print("🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸")
    print("\n\nCentral AI:", central_response)
    print("\n\n")
    transcript_file_to_desktop = "digital_voice_output.mp3"
    voicebox.save(transcript_file_to_desktop)
    is_central_ai_speaking(True)
    os.system(f"afplay {transcript_file_to_desktop}")
    is_central_ai_speaking(False)


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

    print(f"Before number {total_wikipedia_paragraphs}")


    if total_wikipedia_paragraphs > 3:
      paragraphs = paragraphs[:3]
      total_wikipedia_paragraphs = 3
      print(f"After number {total_wikipedia_paragraphs}")
      print(total_wikipedia_paragraphs)
    summary = "\n".join(paragraph.get_text() + "\n" for paragraph in paragraphs)
    for paragraphs in wikipedia_page:
        try:
            for _ in range(total_terminologies):
                current_terminology = technology_count + 1
                technology = get_next_technology()
            if "Other reasons this message may be displayed:" not in summary:
              Centrals_Voicebox(f"Jot down questions on a notepad for me to answer, after I teach you about {technology}")
              Centrals_Voicebox(f"\n\n\t\t     Wikipedia Summary: Terminology {current_terminology} of {total_terminologies}: {technology},,, \n\t\t\t\t............................ {summary}")
            print(total_wikipedia_paragraphs)
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


Centrals_Voicebox("Which one of us should control our study session, me or you")
print('> synaptic pruning')


while True:
    print("Type (me) or (you)")
    study_initiative = input()
    
    if study_initiative == "you":
        Centrals_Voicebox("Certainly, I will lead how we study today.")
        break
    elif study_initiative == "me":
        Centrals_Voicebox("Okay! Take control and ask me questions for clarification. We'll study together.")
        break
    else:
        Centrals_Voicebox("""You need to type either,
                           (me) if you'd like me to rehearse your terminologies, or
                           (you) if you want me to answer your questions
                           """)



def bio_logics_silence():
    engaging_visual = [
        ' cartoon',
        ' mnemonic device',
        ' joke',
        ' visualization',
        ' gif',
        ' meme',
        ' animation',
        ' dark humor',
        ' pun'
        ' comic',
        ' cheat sheet',
        ' infographic',
        ' concept map',
        ' 3D model'
    ]
    if study_initiative == "you":
        ignored_input = get_next_technology()
        selected_image_type = random.choice(engaging_visual)
        url_image = "https://duckduckgo.com/?q=" + ignored_input + selected_image_type + "&iax=images&ia=images"
        # wikipedia = f"https://en.wikipedia.org/wiki/{ignored_input}"
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


def combine_user_input_and_generated_text(user_input, generated_text):
    return user_input + " " + generated_text


previous_generated_text = ""
def Rewiring_You():
    global user_input, previous_generated_text
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    model_responses = []
    with sr.Microphone() as source:
        print("\n\t\t     Central is Listening..\n\t\t🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸 🛸\n\n\n\n\n\n\n\n\n\n\n\n")
        try:
            audio = recognizer.listen(source, timeout=timeout_duration)
            if audio:
                user_input = recognizer.recognize_google(audio)
            
            print(f"You: {user_input}")
            if user_input != "":
                api_response = AI_Mindset({"inputs": user_input})
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
                        Centrals_Voicebox("Hm,, I don't know that one, Timed out")
                else:
                    combined_text = combine_user_input_and_generated_text(user_input, model_responses[-1])
                    previous_generated_text = model_responses[-1]
                    Centrals_Voicebox(combined_text.replace(user_input, ""))

                if combined_text.replace(user_input, "").strip():
                  user_input = ""


        except sr.UnknownValueError:
            if study_initiative == "you":
              Centrals_Voicebox("Moving on..")
              bio_logics_silence()
            pass

        except sr.WaitTimeoutError:
            if not user_input and study_initiative == "you":
                Centrals_Voicebox("Moving on..")
                bio_logics_silence()
            pass

        except ConnectionError:
            Centrals_Voicebox("Connection lost..")
            pass

        except Exception:
            if user_input:
              if API_SECRET == "":
                Centrals_Voicebox("""
                                   Go to https://huggingface.co/settings/tokens and 
                                   get a token. You then must copy paste it inside of my API TOKEN above
                                   on code line number 21. Simply paste it in between the empty double 
                                   quotes for me to answer your questions then restart the terminal.
                                   """)
              else:
                Centrals_Voicebox("ERROR")
            pass

while True:
    Rewiring_You()
