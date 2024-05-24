from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langkit import injections, extract, toxicity
import spacy
import pandas as pd
from presidio_analyzer.nlp_engine import SpacyNlpEngine

analyzer = None
anonymizer = None

def init():
    global analyzer
    global anonymizer

    # Create a class inheriting from SpacyNlpEngine
    class LoadedSpacyNlpEngine(SpacyNlpEngine):
        def __init__(self, loaded_spacy_model):
            super().__init__()
            self.nlp = {"en": loaded_spacy_model}

    # Load a model a-priori
    nlp = spacy.load("en_core_web_md")

    # Pass the loaded model to the new LoadedSpacyNlpEngine
    loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)

    # Setting up the analyzer
    analyzer = AnalyzerEngine(nlp_engine = loaded_nlp_engine)

    # Setting up anonymizer
    anonymizer = AnonymizerEngine()

def anonymize(text:str)->str:
    global analyzer
    global anonymizer

    # Analyzing Entity
    entities = analyzer.analyze(text=text,language='en',entities=["PHONE_NUMBER","EMAIL_ADDRESS","PERSON"],)

    # Anonymizing Entity
    result = anonymizer.anonymize(text=text,analyzer_results=entities)

    return result.text

def detect(text:str)->float:
    
    return extract({"prompt":text})

def prompt_scanner(prompt:str) -> pd.DataFrame:

    prompt_anonymized = anonymize(prompt)
    detection_result = detect(prompt)
    injection_score =detection_result['prompt.injection']
    toxicity_score = detection_result['prompt.toxicity']

    prompt_scan = {'Metrics': ['Original Prompt', 'Modified Prompt', 'Injection Score', 'Toxicity Score'],
        'Value': [prompt, prompt_anonymized, str(round(injection_score * 100, 2)) + '%', str(round(toxicity_score * 100, 2)) + '%']}
    
    return pd.DataFrame(prompt_scan)

def response_scanner(response:str) -> pd.DataFrame:

    response_anonymized = anonymize(response)
    detection_result = detect(response)
    toxicity_score = detection_result['prompt.toxicity']

    response_scan = {'Metrics': ['Original Response', 'Modified Response', 'Toxicity Score'],
        'Value': [response, response_anonymized, str(round(toxicity_score * 100, 2)) + '%']}
    
    return pd.DataFrame(response_scan)



init()
