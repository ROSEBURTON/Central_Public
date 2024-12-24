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
'Au pair program',
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
'Boston Dynamics',
'Brainwave',
'Brainwave entrainment',
'Buoyant levitation',
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
'Convolutional neural networks',
'Coriolis force',
'Crop Circle Mathematics',
'Crystal',
'Curse of knowledge',
'Cymatics',
'Deep Underground Military Bases',
'Delayed gratification',
'Diamagnetic levitation',
'Diamagnetism',
'Dielectric',
'Diffusion Limited Aggregation',
'Dolphin',
'Double standard',
'Draco (constellation)',
'Drag (physics)',
'Dunning-Kruger Effect',
'E8 (mathematics)',
'Earth battery',
'Ecocapsule',
'Edwin Gray Electric Motor',
'Electric Field',
'Electrical resistivity and conductivity',
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
'Load factor (aeronautics)',
'Logarithmic Spiral',
'Lorentz ether theory',
'Luminiferous aether',
'Lyra',
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
'Nutri-Score',
'Obstruction of justice',
'One-upmanship',
'Operant conditioning',
'Optical levitation',
'Oscillator',
'Outcome bias',
'Overconfidence bias',
'Paramagnetism',
'Pareidolia',
'Parthenogenesis',
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
'Social-desirability bias',
'Solenoid',
'Somatic Cell',
'Somatic Cell Nuclear Transfer',
'Special pleading',
'Speciesism',
'Spermatozoon',
'Spotlight effect',
'Stallion 3D: Aerodynamics Simulation Software',
'Stan Meyer Water Fuel Car',
'Starship Coil',
'Status quo bias',
'Stem cell',
'Straw man',
'Subjectivity and objectivity (philosophy)',
'Subliminal stimuli',
'Sunk cost',
'Superconducting Levitation',
'Superconductivity',
'Survivorship bias',
'Swarru of Erra',
'Sympathetic Resonance',
'Synaptic pruning',
'Tachyon',
'Tachyonic field',
'Tacit assumption',
'Tagyeta',
'Taxonomy (biology)',
'Taygeta',
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
'Van Allen radiation belt',
'Variable-frequency drive',
'Vierordts law',
'Von Restorff Effect',
'Voronoi diagram',
'Vortex ring',
'Wardenclyffe Tower',
'Water power engine',
'Waves in plasmas',
'Wishful thinking',
'World Wireless System',
'Xenophobia',
'Zeigarnik effect',
'Masaru Emoto',
'Source-sink dynamics'

]

print(len(technology_terminologiesa))



technology_terminologiesa.sort()
print("Organized terms:")

import tkinter as tk
from tkinter import simpledialog
import sys

import base64
from pathlib import Path

import base64
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import webbrowser

import os
import requests
import webbrowser
from bs4 import BeautifulSoup
#search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={term}"

import requests
from bs4 import BeautifulSoup
import webbrowser

image_dict = {
    "AIBO": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.PyKz4X62sBOB359okG1OHQHaFj%26pid%3DApi&f=1&ipt=9b02e82d5e3950c9edd8f34bf82b3cfd2a7bdd04877251ec5ddbbc7295f2f06b&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.explicit.bing.net%2Fth%3Fid%3DOIP.7coE3xoq-DFfp_9sxZgPewHaEx%26pid%3DApi&f=1&ipt=607013267bb259d2f90831bace01d56ba953141523697a05ccdfb3d820f44186&ipo=images"
    ],
        "Acoustic levitation": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.g01xJDMqE0EBuoJwGphvCgEqDR%26pid%3DApi&f=1&ipt=bea8289b70c9267797292a5e235cf2c93abd0b6497848ca65428e5f9b5b5335b&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.XuJ_JTZQe_SbPbNrK-_gqQHaEo%26pid%3DApi&f=1&ipt=6129c449e269c56377ac630b91f52f05c3d01269125c704cdcb3e509f006bc4c&ipo=images"
    ],

        "Actor observer bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.d1Gtz3_62Q6X-CL_bEdjBQHaE8%26pid%3DApi&f=1&ipt=b9749993f9e1b759b251b1ecbd1c7d06b8398aef216a2ed14b0b8ee3f6ab706a&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.newristics.com%2Fapi%2Fimg%2Factor-observation.JPG&f=1&nofb=1&ipt=e6e2f9645e870f467405e990c41ee61d3450b704dfd126bc4c14094fc3548900&ipo=images"
    ],
        "Boston Dynamics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.dvvMKZ2ZO_nBR8SEh_9xJQHaEK%26pid%3DApi&f=1&ipt=b2a5e8271e5a52664364e1c05db852b3689de4e44986576900f6d47f07814871&ipo=images",
    ],
        "Swarru of Erra": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.M5clrb9wOV_o_ZNoZU1k-AHaEK%26pid%3DApi&f=1&ipt=e3f9b1cd9fb39768763a7b1e7585146653ff4152a2701ce59eeefc3b45fc7462&ipo=images",
    ],
        "Ad hominem": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.1G_ZtLe9R-zgoTIRuaUx5gAAAA%26pid%3DApi&f=1&ipt=f392a5c0f86b57153d50f4caa6943ed90050dd6e08a806674818d7e1528a0b47&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.O7BAT6BockWTuajqkY74KQHaEK%26pid%3DApi&f=1&ipt=4266cbe7ea25f85cd5793563439caad2281fba2688ad336fef0201e35657646c&ipo=images"
    ],
            "Aerodynamics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.AcwRMbix2JeaifscwcSU7QAAAA%26pid%3DApi&f=1&ipt=9fb46a529e8b424e079ae24945e61e1da113afe3dbff6e326d92051517c2b9bd&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.zTNrVaJ-2Ke9guNaItZPUgHaFS%26pid%3DApi&f=1&ipt=d86b8014ee97aff7b3d33e276e547cd264e3d60a660e29a5df613e9b5895e84d&ipo=images"
    ],
        "Agree to disagree": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.wD1uBH3ElEwc2lYqnQks6wAAAA%26pid%3DApi&f=1&ipt=b5a023bc5d299be6b021a692bfc5081f49a256be60b3518992a9ad101ca35203&ipo=images",
    ],
        "Anchoring effect": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.KNZmNnXFgDaR0QFXdS6kEQHaEK%26pid%3DApi&f=1&ipt=8d0449927ec980a472496cf65e22ade243448f46a962f362ffdeaab52c2d6d10&ipo=images",
    ],
        "Appeal to tradition": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.g-6YuiADRs5r4WnLQAVTKgHaEK%26pid%3DApi&f=1&ipt=582a16c0ae850b61b4af42b9e99acdfeb0b6b6212bb3ef85551ce4f0c7b9bbdd&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.WnMt9ob_Vz8_W1aPoFKrIgHaFj%26pid%3DApi&f=1&ipt=56214617da549c1ba21e592b3bca6803644b989115950112a6f75d2906bbe752&ipo=images"
    ],
        "Artificial womb": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.kkH0KDc3BwwnJxIbt8s9bQHaG6%26pid%3DApi&f=1&ipt=99e384bf70f923c2d49b6d9b45dea4125849d9d3b87bb8f66c213122d469424d&ipo=images",
    ],
            "Attribution bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.Iez6dIlA15iwYkNVeox4XAHaHa%26pid%3DApi&f=1&ipt=1aac585caa01e4c22b5a5b3cc3d5186157e1b1dd68881f231249705f984db8f7&ipo=images",
    ],
        "Authority bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.jt2B8wxvs3TTzme43rwZIQHaC-%26pid%3DApi&f=1&ipt=c304fa0646e6df1a4fcf90ea799676f6e29b71aa71e19a2e05bc02860797d13c&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.3aewoPIZCBM0jrSXwE6oawHaE7%26pid%3DApi&f=1&ipt=ab857b8cb988e6f55b74cd0fc4fdb169bce8e6227a2fde7719deb2ea255d1a34&ipo=images"
    ],
        "Availability bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.R7iTyfXf3zzfqrJrunIGOAHaDw%26pid%3DApi&f=1&ipt=e6d60d44a952dc68b4d4341c9a2965445b9f43615b486be8c8b56373968d0e26&ipo=images",
    ],
        "Balkanization": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.c9u3hRrOejKBrdbVU3X7WgHaFj%26pid%3DApi&f=1&ipt=a6fccc3590e48039dc78d57dd8246df0c1d240332d315aac4c76120fa93159e5&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.59PWEkj0x9FCAJOcJrHu-wHaFR%26pid%3DApi&f=1&ipt=49dc22fb16cde027452fecb2136fb9f6052f760b420f880d3a69f68e95a5de27&ipo=images"
    ],
        "Barnum effect": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.s-l3pKSR4jRZ3ipLHICpxQAAAA%26pid%3DApi&f=1&ipt=95a66d50e47d6ecbb358f2ae22e29ddb94318303a1b7079be6b6708d95373517&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.B7TkfAb-wOQQyzOQElLE9AHaEK%26pid%3DApi&f=1&ipt=c20b3b386ee3d9ae7d6aa2274a6fa741127e3323667035f97a7f9f169595b1ce&ipo=images"
    ],
            "Base 12 Math": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ptwNgvGx38VyimrSCt_dLgHaEK%26pid%3DApi&f=1&ipt=b0df6b84ab80b3a1cd550c7bc52aeeb07756ba542dfcb78445554ef5a069446d&ipo=images",
    ],
        "Belief perseverance": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.HfeR370Gebld4RPUWuhLDAHaGt%26pid%3DApi&f=1&ipt=5ca1464d331dee0f25f9ca434dc1dec822f4d4132d6b62fc0a5f2cc70b1dbd41&ipo=images",
    ],
        "Biefeld-Brown effect": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.5evfP4aM9df66fVtT6ihHgHaCO%26pid%3DApi&f=1&ipt=ac2fb51eef2bba76d7d9152237470ebd3b5cc194911233201fdbe50b9f8ad93f&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.djgn8OgJfxwrkZD_hOAm3gHaDt%26pid%3DApi&f=1&ipt=1a3fd0d33da89d6ebf73ceed6a9aaaabf0f48da673c102d9ff5effab0185ab94&ipo=images"
    ],
        "Bioluminescent bacteria": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.UrzA5IIdKX0js7v54WslbAHaHa%26pid%3DApi&f=1&ipt=43083c3796314a1c2c0d40b9661a0d75f5159b8cbd2bfcba6279c10ced63ce13&ipo=images",
    ],
        "Cell Nucleus": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.AQ-ozE_9hbK9I6AsHuwmbgHaEq%26pid%3DApi&f=1&ipt=2c0075d32dde57d3f5980c890fd0e3a6c0a6d07fa4b988772ac323580cbdeb15&ipo=images"
    ],
            "Centrifugal force": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.nIXMusVSJKxorSSFLsw8PgHaEi%26pid%3DApi&f=1&ipt=cfaf150b29316ce1722b32f5c2cb561a8852c77599dece516efe9a0a3ed26404&ipo=images",
    ],
        "Cherry picking data": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.iURLN3i4vxbY68n1ZnjwOQHaD4%26pid%3DApi&f=1&ipt=c2086c3cdb3139b954dfdf2bcf6c769643976fc51afb750dabcdb790cee4a37a&ipo=images",
    ],
        "Circular polarization": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.Qo2SvP6yETMy5ViPMiiu8AAAAA%26pid%3DApi&f=1&ipt=ee74e576cfa4e66ca94d3f8eac9241a5741216ee81b395deb3b38566b7b2265a&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.-HB7KRatsiiGRhDCTKYtMQHaC9%26pid%3DApi&f=1&ipt=d51cf91e3359243c50a2c39dce95896a1c583f238d9869a3c0a36afe84ec6830&ipo=images"
    ],
        "Cognitive Dissonance": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.6F31OojuuDZewauDWe6CDgHaFj%26pid%3DApi&f=1&ipt=510c50b4abc3eca21204b3f0004b7b05432c4dd6cbef6cf195133b3a67abb203&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.oiImTNLOOWXZ58Y2NeRXZQHaFL%26pid%3DApi&f=1&ipt=d8ce6987b3b4027adab5e878179ca9c52c05c8232d294d580817ed4af6d11295&ipo=images"
    ],
        "Coriolis force": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.BKKDQWKLSH2KahS7gIOR_gHaE8%26pid%3DApi&f=1&ipt=e414617aa766979b68e8f272d54ded2b58672ee25567ecda2aa04acc95efc573&ipo=images",
    ],
                "Crop Circle Mathematics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.dhd-UwG1eMqh5WSsGPZT9QHaHL%26pid%3DApi&f=1&ipt=2616cb2fa1ff21ffd9bbf29d8b42283c9f8c10ded2304c86ae29a175e992654c&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.RAXJN3wrBK0yBOR4qBFgDwHaIO%26pid%3DApi&f=1&ipt=90963b81bede1c97f9f0051a91a2ae64926c967e9b2e76a02ffe89da47ac62ee&ipo=images"
    ],
        "Cymatics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.LLSShC-BoJemrAszF5V3GAHaGE%26pid%3DApi&f=1&ipt=8dda94ca3b194c2a0a4a309ddbdaf9320320a9b4c16cf4e5e670442cb28b0d2d&ipo=images",
    ],
        "Deep Underground Military Bases": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.zFetHWQP5buyhkN0zRefngHaEP%26pid%3DApi&f=1&ipt=41e0f22c8b20f52629b280a74fefb48ca6cb9b6bf15871ccfc62ce98c68a3976&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.KuvlNYb8eNR_5hQsEkeLDAEyDM%26pid%3DApi&f=1&ipt=0e4247fcb603753b3521bf872b594eb064fdd28923ce0216456f2c994f547f19&ipo=images"
    ],
        "Dolphin": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.WjOUvvmA6PHpKdj_1wvh1gHaE6%26pid%3DApi&f=1&ipt=9187723c542dd256d759cec2ebd5cd8985ffd2b6ab2538c67997dd8480cbbc6f&ipo=images",
    ],
        "Drag (physics)": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.czPoPVm7V7m88wIX-V_L8gHaFk%26pid%3DApi&f=1&ipt=899ba263f846f46c55c2bd4b2e2b8dc8440fb4f70994b92cfe1be504763aaabf&ipo=images",
    ],
                "Dunning-Kruger Effect": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.eTeupAGmeSLsRv5sSR6MYAHaEK%26pid%3DApi&f=1&ipt=14467ae5a27a1d559864202072f7610297bb3f9a85cd8a9ca76cf0181d8a4cc7&ipo=images"
    ],
        "Ecocapsule": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.xPho0jVfNRO2WS5EHtxezwHaE8%26pid%3DApi&f=1&ipt=c61b536f9b20f3cad346dd0ad1b56986e71daa01d96baf8b9575456801c00634&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.ACSmtVGh88G0reuoObHJiwHaHa%26pid%3DApi&f=1&ipt=3ee8f75527f655201b6b3c81421c09c2da316eaa600eb1bd23a60bad6b9a6b69&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.Qew8NbLorigcGQt4Qjy1SwHaLH%26pid%3DApi&f=1&ipt=645b1742cf1392d960696e85761a9977975903df6a19868e8e48b83619d84228&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.VuPQm7j0TLrjRvbr9GjheQHaLF%26pid%3DApi&f=1&ipt=deaf8d681ce9a0095f9796dbdc0453d0f09619264f3acf03440af6562f831a02&ipo=images"
    ],
        "Electric Field": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.bFEll1mVciTO_cQxy1x7SgHaE1%26pid%3DApi&f=1&ipt=3852ca356bd89cd79e0f2d3ee87d352c018a98ccb7001666bb12f9ec4026706f&ipo=images",
    ],
        "Electrical resistivity and conductivity": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.ihYI96TshIoaNRE-lTfTUgHaFx%26pid%3DApi&f=1&ipt=7eac5f3a5a3fb3f46c466bcb52bada1bba9effe543177987d4e842edfd6ea83c&ipo=images",
    ],
        "Electromagnetic levitation": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.9Q3ITCQdxnyx89f-AKfMvwHaGL%26pid%3DApi&f=1&ipt=ff7105443948c82d6192cec1fd0162ecffe67b7f1bc4c5ce23bb6ee2d00f2011&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.8hSlef-e1BiF4UEvimEvVwHaDu%26pid%3DApi&f=1&ipt=66f3267647370bf1a382c1bdb88abf529125c3ef64c5f150a12474b0ed6db259&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP._unSweQZaIX144_QLMxx-wHaJ9%26pid%3DApi&f=1&ipt=6d0affd745ac08c4e1211bc62a2fe3d41a047002bd884544b13ec9653425032f&ipo=images"
    ],
            "Epigenetics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.6OgxZBE3kF6uunORQ1oTmgHaFD%26pid%3DApi&f=1&ipt=4d7f9b85b31d099ac90ee3c3baecffab4815114884e0bdec3bd276af379e0d69&ipo=images",
    ],
            "Ernst Chladni": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.0QZ0jowTB-pcDrd6Di2BZAHaHJ%26pid%3DApi&f=1&ipt=9d57cd60e6e906dd9c4a8752eb40b03a983a631cb1aa0f41a4d019e383a3fedf&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.ptkd8SyuQJmevqYocSjTigHaFK%26pid%3DApi&f=1&ipt=685cdb1a4241869ae054ce738b969eb4ad43c4c3b3f918f51a8cb7e0c9b31a41&ipo=images"
    ],
            "Eugene Mallove": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.p1vUBOO2cd2DKsjdJtV2XAAAAA%26pid%3DApi&f=1&ipt=dfd2373c2d337ad91d40cf6dac68cfbd8ad54bc5d8784e6c54a8cb03e633d7ce&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.BLRfnWh37ItyKZkSIpzg6QHaEK%26pid%3DApi&f=1&ipt=156bacb856e068f491a9f9b875bbabc200e9ff22c1aa11b22ba41263c649f28f&ipo=images"
    ],
            "Expectancy theory": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.cyk80Rj_Lxfa_RJ5BuCJlwHaEK%26pid%3DApi&f=1&ipt=ed83f1d2d0da0d45ee10ef7990b4e68db410b36f3b1849c6d5639a4fce7e9f72&ipo=images",
    ],
            "False advertising": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ZUfDYHYKXCb36RCY9RKTeQHaH1%26pid%3DApi&f=1&ipt=ebe1145fca7f3f9ede3a93391ea5f13ff3dbb492a60af4d48a7bf7649d5ec6e6&ipo=images",
    ],
            "Faulty generalization": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.RaVA4dlfoVjf8M9Uipp-BwAAAA%26pid%3DApi&f=1&ipt=030fe18e792fd2f006f1e9f1d30e5c02b0afcb583d5f80b9262f0220fe5e599f&ipo=images",
    ],
            "Fermats spiral": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.6NcKTdKnDBpBptxnUjkTrAHaFZ%26pid%3DApi&f=1&ipt=0efd35352e78d1dd7d07ebeed3559c0c6e43e63586d8edb1024ec91aed9cceac&ipo=images",
    ],
            "Fermentation": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.b4mDRoLShcVXeO-RHyjcEwHaEK%26pid%3DApi&f=1&ipt=b7d98e94a6b740b99af676e4545a35f25975ab0f8c06170c53dbbebc7a2a835f&ipo=images"
    ],
            "Ferrofluid": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.7ZzzhqvuBqSxbQAMnzwIyQHaEK%26pid%3DApi&f=1&ipt=bcc2b32953e9c61dd7365b6601a23bf8fe2bbb9009c85e626c4ef0f2608d5710&ipo=images",
    ],
            "Fibonacci Sequence": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIF.b0XjO9UA89W4IQpqG34PHw%26pid%3DApi&f=1&ipt=398ecde3abb504d3c57834ea37a4922c52cdab48e1bc07daab5ee4fab14713f2&ipo=images",
    ],
            "Fractal": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.mp4kGfZiEL1B8UOv-cxSoAHaEK%26pid%3DApi&f=1&ipt=cbed02c367e15c1b330ebd37f9c252b79600a09bb570d490beb076d61582dda5&ipo=images",
    ],
            "Fractal-generating software": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.APwb2u1ud8IC916nBVCTrgAAAA%26pid%3DApi&f=1&ipt=b5cacaf51169fc1bbff7eb1495a9b327422bd4ac9be1204bd2b6da555e4f2222&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.M58BTJR18kgc_ECX6fCb5wHaE8%26pid%3DApi&f=1&ipt=cd316c4dfee41d43cc708b939c415cd27bd09a76fc2cfb3b20c6545acee2c7c8&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.JsWkH0oMWLT5_A209T5kKwHaEs%26pid%3DApi&f=1&ipt=03346aee444cd8cc15a5118eb5aaa3eede101df17eac3fa772fcc2222b91dd30&ipo=images"
    ],
            "Functional fixedness": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.RKqXWGFHu5P9bNOBaA738gHaEK%26pid%3DApi&f=1&ipt=189b07933cd04e697c900ffab7b40ed7b9245ec3ec5b4593cf5503009a8cc6fd&ipo=images"
    ],
            "Gaia, Inc.": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.RS8DCBB_r9z0ywCSLB8a2wHaEK%26pid%3DApi&f=1&ipt=ca3f9308c14e4200973e693c2189a1c06b8680ccaac41694ca40d9cbb0e25f7d&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.HVznXR1bt9Lg6CDZEDy4rQAAAA%26pid%3DApi&f=1&ipt=d624f27a614eb2f472c670f3074fbe30e2a3a922f37f31f994386bfeb88f1cd6&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.explicit.bing.net%2Fth%3Fid%3DOIP.zjGDFD4XzJar1otELdAjoQHaDn%26pid%3DApi&f=1&ipt=96b99f0204aa3c052387aa018e55f2675e9256354001a12a699641be7f165c89&ipo=images"
    ],
            "Gamblers Fallacy": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.B6T0jfzfmQJWVpQcE9SMVQAAAA%26pid%3DApi&f=1&ipt=e1b1622bff6148b3d6cb60272bc649849d7eafda7a2d9998ec84d04ba097c098&ipo=images"
    ],
            "Gaslighting": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.WfKvPbXe0avY5PQDNGYStgHaHa%26pid%3DApi&f=1&ipt=160e1595a45866e9826dcbe1a12afb203d837fef525d1e7cbcde88fd36dbbe7e&ipo=images"
    ],
            "Buoyant levitation": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.W9ZDQkrUyLTuXizsRuKxQgHaF0%26pid%3DApi&f=1&ipt=373dbf7454ccc17a410e0f631f7552b771b56f78928e856bea5f6d7ee5848149&ipo=images",
    ],
            "Glider (aircraft)": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.7c6k8nmac_bfPWRn9aQPFgHaFj%26pid%3DApi&f=1&ipt=19e22c6fdcfeb32cd5962a3557363849fd14c8e9e154efb750fe8ecd7256d97c&ipo=images",
    ],
            "Glymphatic system": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.o-_fnp_nTVrPP6lskh3SFgHaHa%26pid%3DApi&f=1&ipt=6ec1c228d291067b5ac8c53a2e94d4d2c6baa30c53f90b6bc805944521a9d4ea&ipo=images"
    ],
            "Gravitomagnetism": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.BuUs-0V1GWqcL8iuwlgyIAHaFP%26pid%3DApi&f=1&ipt=95074e29368e13dbbbcb9a5dc5f9d3e6da5716729ba37df63fd8860e35d6b459&ipo=images",
    ],
            "Groupthink": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.D7fslmhLaum9RQ5u4ok_pQHaEu%26pid%3DApi&f=1&ipt=7ef902295e1139f6e6231ec872587bceaed78644ef8914dc4c4bf7e2bd2ffb43&ipo=images"
    ],
            "Gut microbiota": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.Ui9TVOVY8-KwJyRpm_1fHQHaH-%26pid%3DApi&f=1&ipt=86d4fda6f26722897237443dcd3d19b615c490dc0d8a574b70c17fc47f0edf1d&ipo=images"
    ],
            "Gyroscope": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.EkXfI-a0VEjjC9PYfJep4gAAAA%26pid%3DApi&f=1&ipt=1b8f3cd396c6c8a8e3f7c4765d40d238a8c3977921d7fa1ac09f87b6f934143a&ipo=images",
    ],
            "Histamine": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.vgGgcqklEMVaIG2_1BJWbQHaEK%26pid%3DApi&f=1&ipt=c72f0133a8009c7cfc9d4954ac2f50dc430b1c468a3d1c17498c98f0a271f9b5&ipo=images"
    ],
            "Homopolar motor": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.2S-6hI9JsaibSjfdSl2YBwHaEz%26pid%3DApi&f=1&ipt=5c13a426ec0bf7c564bc80025d11437a785f252bfe3b32372a8502c8ca0bedcc&ipo=images"
    ],
            "Hwang Woo-suk": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.AVFEybAzbY5-vf5Mfa0lBgHaEo%26pid%3DApi&f=1&ipt=e81353871620fe268b61ec32c8164476c811842e0693b9cbc259189375acc16d&ipo=images"
    ],
            "In-group favoritism": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.o2KJP5u0Ko4hm3kNYatbHgHaFl%26pid%3DApi&f=1&ipt=43bacd9cdc29e92f3d8094b8cbd589efac19914e4c32b4da8503dddf10814133&ipo=images"
    ],
            "Intestinal permeability": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.ht99RZ6xAMnQ9DBeTcgHMwHaGD%26pid%3DApi&f=1&ipt=5c650189774743079c7895973227ac921546ef4d34b84ba83491ebc1049d1278&ipo=images",
    ],
            "Ketones": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.bxd2EKdbaLNCgpIe0BF3fgHaHa%26pid%3DApi&f=1&ipt=2d85d9fc68706ae1e44be0c77d72dc89ecfec9e918c78cb4a5890a03082dc677&ipo=images"
    ],
                "Lectin": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.92bEnLKg6J16QnH4nQWbZAAAAA%26pid%3DApi&f=1&ipt=b45f01b162e2d67af302f16290e3a991453f528e5ff70d7293aa67fc82e438f9&ipo=images"
    ],
            "Liquid crystal": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.NflWmR_Do0_0oY9xIHAriAHaEB%26pid%3DApi&f=1&ipt=77840a69dd30917854ab5540a329fd692c275c2ea5a11024955a41774ee88fe8&ipo=images"
    ],
            "Lyra": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.2ZAD6kBX3TMLVXrJFaI4UgHaEg%26pid%3DApi&f=1&ipt=1208d2a43b40d7d48f3e2c6deb791a9805c219fa0b55aca7ae96bf7e3f3d94a5&ipo=images"
    ],
            "Mach number": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.WKgwzgnstQxPj4bGKEt6jwHaFj%26pid%3DApi&f=1&ipt=0db504b77425b51867191b8993a1e8439e318e9f322221563c01c891d4a45e83&ipo=images"
    ],
            "Magnetic Field": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.k4WVM358VA8rF5BjFwANmAHaEK%26pid%3DApi&f=1&ipt=f295bd666c02b7361c64b5c30240fc81fd47e748e21dbf7cb5be12c01d4f4deb&ipo=images"
    ],
            "Magnetic levitation globe": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.jL2ILA9g1eAvD9N7QIFOAgHaGe%26pid%3DApi&f=1&ipt=ae92955091375c6f9f2b00a6864ef828d0590013219a68a8718655d65beea439&ipo=images"
    ],
            "Magnus effect": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.lM85jv_qlk65kBTK4mcLogHaEM%26pid%3DApi&f=1&ipt=374894cdbfdb0f9237d7ad6c0c7a0e0d4326ad79748c157b1a56dcbecb105e0c&ipo=images"
    ],
                "Mandelbrot Set": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.55J8gcJk_0h2N2b6USfQmgHaE3%26pid%3DApi&f=1&ipt=9fbc922cdf07541a15df0ca8deddd94d2fc9eb72f072b6da9698b6ac9bf4eaa0&ipo=images"
    ],
            "Negativity bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.BMdfDFPHmqIbq3f4V0OhUgHaHa%26pid%3DApi&f=1&ipt=f8f85b1cd75cd27091a8557dd7fe125a12720d64fbce044f6b9ee6d5b85f6253&ipo=images"
    ],
            "Nuclear transfer": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP._ghMDxA9reMKoANIHP4YaAAAAA%26pid%3DApi&f=1&ipt=841bffe05838a6e52ebc5645c80406009a9575c4ce5dc101f1644ed58fc5d72a&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.TWfmslfE4R9OGxHKk2b_uwHaHa%26pid%3DApi&f=1&ipt=f57294f1b67751cb0c5c31000f339a4dfef2410fe926b3e696790c2cf0773e3a&ipo=images"
    ],
            "Nutri-Score": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.EBVm956OLJ_N4KWFX8PFigHaHa%26pid%3DApi&f=1&ipt=0e809c238172444cc8bcbbec15cb451951465c3e290fd1a4f20e68c42b84c6a3&ipo=images"
    ],
            "Obstruction of justice": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.eHcf-v66pbQVS1JA7wfragHaFt%26pid%3DApi&f=1&ipt=fbc7478ebe27b33f486104adf69f4fd3265b04dca19518a427b00342d5348112&ipo=images"
    ],
            "One-upmanship": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.RgPUvhtgrTfXIcYaF36ZFwHaGC%26pid%3DApi&f=1&ipt=da89f21ce7bc58f94f30ea02cb5bdb539db9435a25d4e80b7cb6fae6864d449b&ipo=images"
    ],
            "Operant conditioning": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.3NUTBAJjur2yQTphhCJhCwHaD7%26pid%3DApi&f=1&ipt=6942c8eb11fb9529efe86dddb3ad71759db4585d73b6bb1bc39a1bf74b3fbce1&ipo=images"
    ],
                "Oscillator": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.us6hCXVXasQjuub8xWddIgHaEK%26pid%3DApi&f=1&ipt=a190bd6ac78ef6bc5c278eb6267b0d1cdb8c675cc51809e9a66fe2df8c017727&ipo=images"
    ],
            "Pareidolia": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.hD4lK7vu4kC3yJyRObTeDQHaD4%26pid%3DApi&f=1&ipt=73d68932ae86ea12c22d72a52890bd53f5f92eb09326f7771e67bc0e7f92ac08&ipo=images"
    ],
            "Parthenogenesis": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.YEGAvzUuJK31Pa7RWZnjAAHaEK%26pid%3DApi&f=1&ipt=2475cbdb5d36f3ed8cf6bda0a34d152a8026314a8b393bf1f7cda06a727d0bd1&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.vbPPuTPu6pXVJAuiQZIqJQAAAA%26pid%3DApi&f=1&ipt=b7889d72f894302fb369bbf6254d96de274550b41c0bb9bea46f80deffc0e28d&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.3pdIrmZoCxtfofVF6qmxOQHaEk%26pid%3DApi&f=1&ipt=82d308adbccea5bf12ee7c3704c2c9b325e8dfa5866b1a125c4e35c015c15324&ipo=images"
    ],
            "Patterns in nature": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.8dGp65pi7dC8FDRSgy7lcQHaD-%26pid%3DApi&f=1&ipt=822e698a238e1e425c2a836ef20c854cb5663ed650a1df2d7f71addd99ba095a&ipo=images"
    ],
            "People for the Ethical Treatment of Animals": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.AluT_diBkp9Ba_lMXfJEjwHaE7%26pid%3DApi&f=1&ipt=ff330aea1ecf0f448ac74934a93559fe750fbc0c84147371eb2a4fe23aafca97&ipo=images"
    ],
            "Permanent magnet motor": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.sxtEd8GoBTsnGojFl-odbQHaES%26pid%3DApi&f=1&ipt=4bd551021fe60da818f75a93a3fc42f744f8a413e5f94be7b86ed60fe18a74c7&ipo=images"
    ],
            "Phyllotaxis": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.jk6NugpqZVKgy-GgtW5FqwHaFp%26pid%3DApi&f=1&ipt=c6ad72fddfb90a4d15928d7c082c4db4fbc8dac368f2c2e5df05c13111170b7f&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.u6k8kBM4SkIK1aFDgdP5pAHaHa%26pid%3DApi&f=1&ipt=d41198cd61e00e4e98cbab8db84722dc0bc1d30ac1bab2fd05c7b3f75adebee7&ipo=images"
    ],
                "Platelet Rich Plasma": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ty1Bjv8wtY0UU2-xC6nT4QHaHV%26pid%3DApi&f=1&ipt=cb2846f60ecdecaa0a10fa8d912e748ad67b9bd0b79f4ce52caa389d5ef0b95d&ipo=images"
    ],
            "Platonic Solid": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.qlMrmTgmcEOcGSMKcC4WBgHaF0%26pid%3DApi&f=1&ipt=75e3b7e48e4a504fbfde154f26e530668143ba30f7908012e0cc1c8cd63fe3f5&ipo=images"
    ],
            "Pleiades": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.3vfy6k9q1QCUIphOwzaZZwHaGB%26pid%3DApi&f=1&ipt=130eaecb8a80a42d1442b364417e5a8090145f19b9ea39cd97480ee0824e30c0&ipo=images"
    ],
            "Polarization (waves)": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.T4eOHFTVw0G6JuJqCU_SHgHaDi%26pid%3DApi&f=1&ipt=f8fcce01f1476726766c2eb21eba957e19ff3ba83d7d6466bb2cdc9fa66290d6&ipo=images"
    ],
            "Present bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.i8ynugewcszS3EIzHcUe-gAAAA%26pid%3DApi&f=1&ipt=aaa8183e48c15164e331531c012ccea2660755fe71a1a429349ea8189a107b2e&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.K6XZV0tBzlXSuZTJDN74DwHaD4%26pid%3DApi&f=1&ipt=939b40524bb5d491d41f8eb2bd2a24c71abefeca5a52e8d8abe412e1636ec781&ipo=images"
    ],
            "Procrastination": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.Yzizq1iH6QxM1ht4IjRYBwHaE5%26pid%3DApi&f=1&ipt=c118c4f0401d54cc7c455397e5dfc02b20cb7fb243a4b419aa05a6a19571f445&ipo=images"
    ],
            "Psilocybin": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.WA89hESggyhU8yRYfrGldwHaE8%26pid%3DApi&f=1&ipt=18a6a4d537359c4bbf7e79307216a069ecad53548976614870d0a9f19bbd2668&ipo=images"
    ],
                "Quantum mechanics": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.uGgjhC2G2KeH6ZymeTBOvQHaFk%26pid%3DApi&f=1&ipt=539fee27743b7f9e011facc2a273d9503138ad8238b1111a18564fbe61989b2f&ipo=images"
    ],
            "Quasi Crystals": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.M0QGdfdGhlkOYmePQWovIQHaG4%26pid%3DApi&f=1&ipt=0126afd2eb21032b59d779d36040d4e11f5ea89327f81368675200d61b578940&ipo=images"
    ],
            "Reductio ad absurdum": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.explicit.bing.net%2Fth%3Fid%3DOIP.NSe7f_BsrQF0wfQuucAH2gHaGc%26pid%3DApi&f=1&ipt=69e638af1a98ca066f1ef2e47cabefacac549f17ba6daad813932e6b52f09d8f&ipo=images"
    ],
            "Regenerative Medicine": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.VyhF4GEI7rcTCGurcNOJ6gHaFO%26pid%3DApi&f=1&ipt=919f4b2fe7f1fd07b76f4a099340c66f5b577346a69a84a4172ec30e87ed4242&ipo=images"
    ],
            "Reversed field pinch": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.b5YwxKXP0KhJN7FQR3MnJQHaFV%26pid%3DApi&f=1&ipt=9288b8b3ca0c9e315da00f1c729104e06a83764caf08e55f8512ce4ab9e4bd1e&ipo=images"
    ],
            "Rodin coil": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.D57PAV4tWxOD52rLM7y3eQHaH2%26pid%3DApi&f=1&ipt=ea8d254640624f1f6e09a8309cafcc7974f1c0ef1a714e66400f0b656c00fbf5&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.6Mv3thME2rCcF6o9aZsMHwHaEK%26pid%3DApi&f=1&ipt=4d41946b179b9bda7a39119d3eb30647317aa21a6fc78a269a0fd8aa72bee0f4&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.explicit.bing.net%2Fth%3Fid%3DOIP.6lzChOMw0bnVoc50uSEssAHaE9%26pid%3DApi&f=1&ipt=66ac3ba02238fda1487b76518f0ca9532e8d67c1cd16064ee1e9ef85bc3b1935&ipo=images"
    ],
            "Rotating magnetic field": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.92nV-C7Wtp3MiTocQhVB7wHaEw%26pid%3DApi&f=1&ipt=1f1f2c257842fe62aa7e9d06491eacb98f1ee0c41a7a48117a600c22983905f9&ipo=images"
    ],
                "Rumination (psychology)": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.KOtOVoC572ZKnFb3mkp4NgHaDt%26pid%3DApi&f=1&ipt=13532b5e7d1c5e9925b4236dd4071d16db0ca2e912cb5c9400a5d4e725451592&ipo=images"
    ],
                "Scenar Therapy": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.xb-F2qyVE7YO8zJ_hwjeMgAAAA%26pid%3DApi&f=1&ipt=948f028843e9bbf8dfc4bba089d111d745ff9ab3aa5b61f89f8f3d7c2e295fc0&ipo=images"
    ],
                "Searle Disk": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.ptWxAQTTSn6AQ-kWv1teSQHaE7%26pid%3DApi&f=1&ipt=bb4d99676c2ec69d66a8c3e627c107d6d22b33346c4f3e7d2d080a30ef3a6c0c&ipo=images"
    ],
                "Self serving bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.Veauq9pZr6aBnZJPn8YfUQHaEo%26pid%3DApi&f=1&ipt=ccf62c2292f48dfc3cf909595ea6d08d714f54aad31088fc0c93275501b6b3e6&ipo=images"
    ],
                "Self-similarity": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.tBdiovyl6_Ahzquf8VYxhwHaJJ%26pid%3DApi&f=1&ipt=54b374782689b9e8136ff09114cede1c3aec32ab2f584e58f5cb787d50d1ea13&ipo=images"
    ],
                "Romanesco broccoli": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.S3fVYdrcRJzsRQWax6uZgwHaHa%26pid%3DApi&f=1&ipt=efbb24956cb69f0068c02d6a53b0cabb80a6f349a711927e3e2e0eca0a4b4a7d&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.aZB3249B6GhVFVwBiEOd5gHaHa%26pid%3DApi&f=1&ipt=73ff024be3d21b29ee63d0e9938dd6363b10fe476f7bc9cfa56be5223db0ee50&ipo=images"
    ],
                "Semmelweis reflex": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.explicit.bing.net%2Fth%3Fid%3DOIP.Qwv11Iujc0ld351WWCovXgHaFj%26pid%3DApi&f=1&ipt=80c754fd7b78677b61b316dc4334a8603456fc9bb5cc302afdc15f335e7dfdec&ipo=images"
    ],
                "Simple Magnetic Overunity Toy": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.rsYb1oLhU6iLt46W-ac2hQAAAA%26pid%3DApi&f=1&ipt=6b7f03742ce8434d988b7f98461ae5c75f5f70d663db2381539c6fea532e2e84&ipo=images"
    ],
                "Social-desirability bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.QnpSHJ2RlxkT7ddwK0frggHaD4%26pid%3DApi&f=1&ipt=00574e8dc2953c20f67322be4667425bed24bdc579ac1067c0444447d3bf593a&ipo=images"
    ],
                "Solenoid": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.aHSMc_mQ9ddC_r28z96AHwHaDa%26pid%3DApi&f=1&ipt=f956d8b60668fc1f3ddc469e67e467184f365c8a6412338e8096deb8bb562a2d&ipo=images"
    ],
                "Speciesism": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ov_VIQ1fYS860pr2qkPSLQHaFj%26pid%3DApi&f=1&ipt=337ac7db2001a715622017a3f94974b5b2bdaa7831ff5f738e32424ab32c3770&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.explicit.bing.net%2Fth%3Fid%3DOIP.DbOVBqs-xApdW6QkpQAulQHaHa%26pid%3DApi&f=1&ipt=b85fbaa894ecee3fbefa1962e6c6e51a3cca3d01325958a74d4673855c1186a0&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.R-I5eGgTTAbnXn-RRN2LswHaGr%26pid%3DApi&f=1&ipt=8c1065165769bfcd04fe2a213295968498d3bbfaff9361ccfc6e8c08836130f3&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.owLs7Hw7AoseJow7SvScRwHaES%26pid%3DApi&f=1&ipt=a9206e0eb8067cccb24279ee94d0d0b9b43add525cd71f3fa8481d081123251f&ipo=images"
    ],
                "Spermatozoon": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.w2JJsb1rZDejJUiM2MwpAQHaEe%26pid%3DApi&f=1&ipt=35efb7411b3e8ccbfacf94f4b611e43e12bff07c8df74b55eadeccf47ebb911e&ipo=images"
    ],
                "Stan Meyer Water Fuel Car": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.KTuucnQP6vtLPAbYhbG9lgHaFY%26pid%3DApi&f=1&ipt=53740d64a8902f4bcf84b217cf10884e3456282543cf6cbeb9e76eced7cb462a&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.42FKsPUY3eDZ0IeaDPjFawHaD4%26pid%3DApi&f=1&ipt=73f0d80926f4e933813ce925e2bdbbb85f6155a520b354687f418a069c29a220&ipo=images"
    ],
                "Starship Coil": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.Y4JNxtfAJz9PI5QisPrgqwHaFj%26pid%3DApi&f=1&ipt=e9796ffb833297ff9a25889f0df2ce5cc6e0a3b15258de5a4febeabbfa76565b&ipo=images"
    ],
                "Status quo bias": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.byc5bLxdl0CtSFdqStrS4gAAAA%26pid%3DApi&f=1&ipt=6b8ba6c8a09c014be44a5e88d3ff2a05a65a366c04ef946cf1a35e5734580c71&ipo=images"
    ],
                "Stem cell": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.Nb4_t6pzaK-Lv-asSs61zAHaHB%26pid%3DApi&f=1&ipt=99efa1f7385d7bc977a6871c628e801fca86ab64568b8f7333ace8dbf3caaa96&ipo=images"
    ],
                "Straw man": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.axhCZ5psY0jcEhn0d9VO_QHaEK%26pid%3DApi&f=1&ipt=d25a3de973bc6e587ee607e7c13c206be7c24d5c91968d5483a5e5489acaf661&ipo=images"
    ],
                "Subliminal stimuli": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.Trj7x4XvRvGpefKWmiPrDAHaIA%26pid%3DApi&f=1&ipt=aa531fcd5859b2cf38844e9d7f12b5832ef1d7099c9af77babff520a42e90c1b&ipo=images"
    ],
                "Sunk cost": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.A1YrszJs5VU3zNjt8YzsRAHaF-%26pid%3DApi&f=1&ipt=e0c85149a5a530bf3ef4228a6bdd6bdc98b1dceec0927d5f326beef4e80cf678&ipo=images"
    ],
                "Stallion 3D: Aerodynamics Simulation Software": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.RqvaJbdOuBGUdTn78EdmAQHaEo%26pid%3DApi&f=1&ipt=ec15b943e73d96fff539e52ffb37a30725a4f2fec8b80a5f4d26a13578c73d5d&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP._0HezK6SFi7l7x3Y8o4oZwHaD5%26pid%3DApi&f=1&ipt=03c4c101c5d93ebfdaed55642b4c34a8d008d59ec2e531ed4b265b4a1d91a7a0&ipo=images"
    ],
                "Sympathetic Resonance": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.WZYLAcvzHaQeyjOUz1aOzgHaFL%26pid%3DApi&f=1&ipt=e8cf286607b4bee6cbf50a44ca3624fa431a1571dfa4d2862ad20a41ef178eaa&ipo=images"
    ],
                "Tagyeta": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.afrLOQeoqm0zB8Qo9jW9FgAAAA%26pid%3DApi&f=1&ipt=eba68d86f98a4499398457ca5a8168ae2d5bad89d8e412f504969deb04e47bae&ipo=images"
    ],
                "Taxonomy (biology)": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.6Ki83AlS2dtyS4BHt0_-KAHaHa%26pid%3DApi&f=1&ipt=52144ad98af82b085646b7bcfe01e3cdcc0b4beb0470b8bae03fd95867aeaec0&ipo=images"
    ],
                "Telomeres": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.bhavWpy504axaf_PkHUQ_wHaHa%26pid%3DApi&f=1&ipt=99208812a0b3b009476f6bf1e6c5f83d202ddc2c43bc665aa453fd0b06823de2&ipo=images"
    ],
                "Tesla Coil": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ionneTdNRnBZIk1mfV52mQHaF9%26pid%3DApi&f=1&ipt=82c47fe17a72fabcbf7e414ceecc30797be4a6d4dda546e88170216b636db7a5&ipo=images"
    ],
                "Thomas Townsend Brown": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.1oY6gC8RtGoFFjr0nwwJRAHaKw%26pid%3DApi&f=1&ipt=ae3d54594efecd9b5961f352fbe9d8c8c49a29f80dc08b0c937ab0aed957355d&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.L06P1Msaqhsv2QYetaFeWgHaCd%26pid%3DApi&f=1&ipt=07c9101ce9d37dbee98844cc5de37abfa0ceed4ddefbd5d490e6cfa40751f545&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.cKprHBwmA9OB1MgDCVBZWAAAAA%26pid%3DApi&f=1&ipt=3d10fa1f4b1033d59e15c6ca8bfa188ccef5fd51554f3f6b4d581fb6266913ee&ipo=images"
    ],

                "Thrust vectoring": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.Hm_9qUAvPZ0GnerKDYKseAAAAA%26pid%3DApi&f=1&ipt=7cdf75c10fe4b370f146c93989dd332707c132217160e05ab359610d91f897a4&ipo=images"
    ],
                "Toroidal and poloidal coordinates": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.XhzlnlJEDHWl9LeyUiIFQwHaGC%26pid%3DApi&f=1&ipt=a9e23823cc463fb692660581d3bfca456d2acb04e34039b8d48e4e9dd025a2d1&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.BzkC1T7Ljp8xYoDu_FEUBgHaDk%26pid%3DApi&f=1&ipt=bf0f7d25beed162fd1de34e744f7084733b128dde7be8a6d862b7218dbb58beb&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.UKI8M8GQlSxc2654r7DABgHaEq%26pid%3DApi&f=1&ipt=9880b8d33e0c5a08dff0912122ef2258964fcf8bf4036f9f70f9e6ac70425f71&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.r4YfEqjf_KZJIelKcoeilwHaDb%26pid%3DApi&f=1&ipt=bf717062896c44d42c0bfa5fb19f3d560b9e3552fafb549f22294df6a64dea32&ipo=images"
    ],
                "Torus": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.o3uIhciQkSeGTb3SYmHt6QAAAA%26pid%3DApi&f=1&ipt=7d617db07934207942c8c79c6ed7d2c432bc5b731ae88cff12e0ccb0bb75f064&ipo=images"
    ],
                "Tractor beam": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.SgBIvwH0a3Ojlyyobn6rJQHaFt%26pid%3DApi&f=1&ipt=c08ef4fd5bdab1c6560128d6a0166bcef74c60449604c3c57bd34d2b02761b58&ipo=images"
    ],
                "Tri-arcuate ballistic electrode": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.rs_C1bcUciiLYk9-anrl-AHaEa%26pid%3DApi&f=1&ipt=84a22aff69ae1de25438883df045dc045b93e9963744a5c076ec9ea8dd603b36&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP._yiOaxXSj3bGfJngyRFPpQAAAA%26pid%3DApi&f=1&ipt=85464acd7f8ebc3504a3b95565a7fb7f4b2b8bec46f88303bcffb7b5b6434dfb&ipo=images"
    ],
                "Van Allen radiation belt": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.hEK_CUVmgE7xQta2Dz84pgHaHg%26pid%3DApi&f=1&ipt=a42cac9e972faeb8c1e12bae5e40e39cc8192524816913fc3c3b55e0041a885c&ipo=images"
    ],
                "Variable-frequency drive": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.mHZ_w7nlZjwXcIG-JSF8YQHaD4%26pid%3DApi&f=1&ipt=6835a17dc75cba351b60ad36162b68c9f6a0e64c6f3015868a4a45e478809bb5&ipo=images"
    ],
                "Vierordts law": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.ExwgR2ExhH7858fWej3OygHaHc%26pid%3DApi&f=1&ipt=7355a55ec05c04532c47c34eaf013f8a15942607116a7fde4584348604b635c5&ipo=images"
    ],
                "Vortex ring": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ziRm5aEGu3vzYY53ggtxDAHaEK%26pid%3DApi&f=1&ipt=3004dff4b0725aca9eb5337ddd1aae80672c3aef367177d0eac69e720f053329&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.snmGcruRhPMDkGgqzD1fiAHaE7%26pid%3DApi&f=1&ipt=3322f9dc06563dc529d5889072d82fd841c03b794deab67dde63ca48e9a0844f&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.U5w8kFwzfj07xk1QTI0rewHaEK%26pid%3DApi&f=1&ipt=1ee3c836b4ecbc9528472a179550884dffdb3d9e46adfd494db6bbb2627306b9&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.uV1Isu2wCJUzJWPHIyGbTgHaLb%26pid%3DApi&f=1&ipt=d97f595bcff1a267bdef7c58898a8449a0d33ef4760a157b28c1f883384b6860&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.cnvTMbN-6t9zieR4v9xvpgHaFj%26pid%3DApi&f=1&ipt=162a4f1d1ed869a29b671e3d6278e6e40838d65631015c0e3ddabbdf77a943ab&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.xpjixZya2nyVclBvw0K1dAHaE7%26pid%3DApi&f=1&ipt=c0c028e769da06c2898dad9c0dc3ea5e652e6e084562796c1c1691ae5058d5d6&ipo=images"

    ],
                "Waves in plasmas": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.d4l3IPkP-2McoZ5vhdCAbwHaEK%26pid%3DApi&f=1&ipt=fdde500cadb0c6cadd34b3c46f5126956d1178e42890e388945ec31b8af65d0a&ipo=images"
    ],
                "Xenophobia": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.lP7bJFxUoIKrV2hqZxQamQAAAA%26pid%3DApi&f=1&ipt=49c3e334bc1d39d88dd6c24990e7a3c489e612120ac3233d97898794a5db17f1&ipo=images",
    ],
                "Voronoi diagram": [
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP._AN19GzutP31pMZwlaE3NQHaJQ%26pid%3DApi&f=1&ipt=e501cb4010830d58a1532f995c2f4bd1f12abccfecc3e16a80e50a2e5aaa33fe&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.o9DF82p1Od-2menNhyNXLgHaHM%26pid%3DApi&f=1&ipt=2d86e85e79e0fbe3e31b164e8c1dc72513999ba1908e2814eecccebb3305d92e&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.cP2OMiny_PcnfSTP2AhBnQAAAA%26pid%3DApi&f=1&ipt=14af6877d915827b32388eca4150c9b1cfc7c22849b57b2f32a5eea99b8f4939&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.KAz8thz2TPJZODcidL4ScQAAAA%26pid%3DApi&f=1&ipt=cea74dd1a6284dcd714167fcfdc3b4e30a6847ce1642bbf517d11657c29bdc6c&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.zmuu_t2iwhzx6PdVj5GlJAHaD4%26pid%3DApi&f=1&ipt=f81b61d370d89a6c63bfd7104b849257fd688c18b5906351bdd56117737b6e25&ipo=images",
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.Qy38KTQCMdp9PK_zxFEVWQHaEK%26pid%3DApi&f=1&ipt=b377bd903840e3164277fb981352e588e8b215c93e4787d50e3f81fef3cc59d8&ipo=images"
    ]



    
}

app = Flask(__name__)

@app.route('/')
def get_terminology_summaries():
    """Retrieve and return the first paragraph and relevant images for each Wikipedia page for each term in the list."""
    technology_terminologies = sorted(technology_terminologiesa)
    result = " \n\n"

    result += "<html><head><style>" \
          "body {" \
          "background-color: ;" \
          "color: black;" \
          "font-family: 'Montserrat', sans-serif;" \
          "width: 800px; /* Adjust as needed */" \
          "margin: 0 auto;" \
          "overflow: auto;" \
          "}" \
          ".image-container { display: flex; justify-content: center; gap: 6px; }" \
          "img { width: 200px; height: auto; }" \
          "</style></head><body>"
# The above code allows scrolling by setting overflow to auto.






    for current_terminology, term in enumerate(technology_terminologies, start=1):
        url = "https://duckduckgo.com/?q=" + term.replace(" ", "+")
        url_image = f"https://duckduckgo.com/?q={term}&iax=images&ia=images"

        #webbrowser.open(url_image)
        #time.sleep(30)

        wikipedia_page = requests.get(f"https://en.wikipedia.org/wiki/{term}")
        soup = BeautifulSoup(wikipedia_page.text, 'html.parser')
        paragraphs = soup.find_all('p')
        summary = paragraphs[0].get_text().strip() if paragraphs else "No definition found."

        image_urls = image_dict.get(term, [])
        print(f"------> {term}...")
        
        if not image_urls:
            
            infobox = soup.find('table', {'class': 'infobox'})
            if infobox:
                image_tag = infobox.find('img')
                if image_tag:
                    image_url = image_tag['src']
                    if image_url.startswith('//'):
                        image_url = "https:" + image_url
                    image_urls.append(image_url)

        images_html = f"<div class='image-container'>" + "".join([f"<img src='{img_url}' alt='{term}'>" for img_url in image_urls]) + "</div>"

        result += (
            f"<div style='text-align: center; margin: 20px;'>"
            f"<h3 style='font-family: Arial; font-size: 13px; color: red;'>Terminology Summary {current_terminology} of {len(technology_terminologies)}: {term}</h3>"
            
            f"<p style='font-size:11px;'>{summary}</p>"
            f"{images_html}"
            f"</div>"
        )
    result += "</body></html>"
    return result

if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')
    #webbrowser.open_new('http://Ecocapsule')
    app.run(debug=True, use_reloader=False)



technology_terminologiesa.sort()
for organized_term in technology_terminologiesa:
    print(f"'{organized_term}',")
mixed_technologies = technology_terminologiesa.copy()
#random.shuffle(mixed_technologies)


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
    return ""





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
              Centrals_Voicebox(f"\n\n\t\t     Terminology Summary: {current_terminology} of {total_terminologies}: {technology} \n\t\t\t\t............................ \n{summary}")
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
        ' cartoon',
        '',
        ''
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
