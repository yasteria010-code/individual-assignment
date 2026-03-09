import streamlit as st
from PIL import Image
import time
from transformers import pipeline


# function part

# 1. image to text (captioning)            
def img2text(image_path):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = image_to_text_model(image_path)[0]["generated_text"]
    return caption

# 2. text to story (limit to 50-100 words)

def text2story(caption):
    story_gen = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = (
        f"Write a short summary story (50-100 words) for children "
        f"based on this image caption:\n{caption}\n"
        f"The story should be simple, warm, and easy to understand."
    )
    story = story_gen(prompt, max_length=120, min_length=50, do_sample=True, top_p=0.95, temperature=0.9)[0]['generated_text']

    # to ensure story length is within 100 words
    words = story.split()
    if len(words)>100:
       story = " ".join(words[:100]) + "The end!"
    return story

# 3. text to audio
def text2audio(story_text):
    tts = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = tts(story_text)
    return audio_data


# steamlit app 

def main():
  st.set_page_config(page_title="Image to Audio Story", page_icon = "📖")
  st.title("📖 Storytelling App for Kids (3-10 years old)")
  st.write("upload an image and enjoy a fun story with audio!")

  uploaded_file = st.file_uploader("upload an image", type= ["jpg", "jpeg", "png"])

  if uploaded_file is not None:
     # save uploaded file locally
     bytes_data = uploaded_file.getvalue()
     with open(uploaded_file.name, "wb") as file:
         file.write(bytes_data)
    
     # Button interaction
     if st.button("Click Me"):
        st.write("🎉 You clicked the button!")
     
     # display image
     st.image(uploaded_file, caption = "upload image", use_column_width = True)

     # stage 1: image to text
     st.info("🔍 Generating caption from images...")
     caption = img2text(uploaded_file.name)
     st.write("**Caption**", caption)

     # stage 2: caption to story
     st.info("✍️ Creating a short story...")
     story = text2story(caption)
     st.write("**Story:**", story)

     # stage 3: story to audio
     st.info("🔉 Converting story to audio...")
     audio_data = text2audio(story)

     # play audio
     st.audio(audio_data["audio"], sample_rate = audio_data["sampling_rate"])
        

if __name__ == "__main__":
    main()
