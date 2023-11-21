# See the demo with voices:
# https://huggingface.co/spaces/Matthijs/speecht5-tts-demo

from datasets import load_dataset
from langchain import PromptTemplate, LLMChain, OpenAI
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


# Models: https://huggingface.co/tasks

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


filename = "result"


# https://huggingface.co/Salesforce/blip-image-captioning-large
def image2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image_to_text(url)[0]["generated_text"]

    print(text)

    return text


# llm
def text2story(scenario):
    template = """
        Generate a short story based on a simple narrative.
        The story should be no more 20 words.

        CONTEXT: {scenario}
        """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    llm = OpenAI(model_name="gpt-3.5-turbo-1106",
                 temperature=1,
                 openai_api_key=OPENAI_API_KEY)
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)

    return story


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

    sf.write(f"{filename}.wav", speech.numpy(),
             samplerate=16000, format="WAV")


testScenario = "the rabbit near the house"
testStory = "A woman finds the painting in an old attic and is transported to the place in her dreams."


st.set_page_config(page_icon="🦚", page_title="image to audio story")

st.header("Push image into audio story")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

submitted = uploaded_file is not None
if submitted and OPENAI_API_KEY.startswith('sk-'):
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)
    st.image(uploaded_file, caption="Uploaded image",
             use_column_width=True)

    with st.spinner('Storytelling...'):
        # TODO scenario = image2text(uploaded_file.name)
        scenario = testScenario

        # TODO story = text2story(scenario)
        story = testStory

        story2speech_with_transformers_microsoft(story)

        with st.expander("scenario", expanded=True):
            st.write(scenario)
        with st.expander("story", expanded=True):
            st.write(story)

        st.audio(f"{filename}.wav")
