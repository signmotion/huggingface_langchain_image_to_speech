# See the demo with voices:
# https://huggingface.co/spaces/Matthijs/speecht5-tts-demo

from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset
# from IPython.display import Audio
from langchain import PromptTemplate, LLMChain, OpenAI
# import os
import requests
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from TTS.api import TTS


load_dotenv(find_dotenv())

# Models: https://huggingface.co/tasks

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


filename = "stream"
image_file = f"source/images/{filename}.webp"


# https://huggingface.co/Salesforce/blip-image-captioning-large
def image2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image_to_text(url)[0]["generated_text"]

    print(text)

    return text


# scenario = image2text(image_file)


# llm
def text2story(scenario):
    template = """
        Generate a short story based on a simple narrative.
        The story should be no more 20 words.

        CONTEXT: {scenario}
        """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)

    return story


# story = text2story(scenario)


# Use dataset and transformers directly.
def story2speech_with_transformers_microsoft(story):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=story, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write(f"output/{filename}.wav", speech.numpy(),
             samplerate=16000, format="WAV")


def story2speech_with_inference_api_microsoft(story):
    # TODO Internal server error.
    API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query({"inputs": story})
    print(audio_bytes)

    # Audio(audio_bytes)

    # audio_segment = AudioSegment.from_file(
    #    io.BytesIO(audio_bytes), format="mp3")
    # ogg_data = audio_segment.export(format="ogg")

    # with open(f'output/{filename}.flac', 'wb') as file:
    #    file.write(ogg_data)


# XTTS is a text-to-speech model that lets you clone voices into different languages.
# https://huggingface.co/coqui/XTTS-v2
# https://huggingface.co/spaces/coqui/xtts
def story2speech_with_inference_api_xtts(story):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=story,
                    file_path=f"output/{filename}_xtts.wav",
                    speaker_wav="output/stream.ogg",
                    language="en")


testStory = "A woman finds the painting in an old attic and is transported to the place in her dreams."

# story2speech_with_transformers_microsoft(testStory)
# story2speech_with_inference_api_microsoft(testStory)
# story2speech_with_inference_api_xtts(testStory)

# print(f'The speech file saved to the folder `output`.')


def main():
    st.set_page_config(page_icon="🦚", page_title="image to audio story")

    os.makedirs("output", exist_ok=True)

    st.header("Push image into audio story")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded image",
                 use_column_width=True)
        scenario = image2text(uploaded_file.name)
        story = text2story(scenario)
        story2speech_with_transformers_microsoft(story)

        with st.expander("scenario", expanded=True):
            st.write(scenario)
        with st.expander("story", expanded=True):
            st.write(story)

        st.audio(f"output/{filename}.wav")


if __name__ == '__main__':
    main()
