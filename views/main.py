# Importing required libraries for Streamlit app, image generation, and file handling
import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv


load_dotenv()
# Setting up Hugging Face token from environment variable for secure access
hf_token = st.secrets["hf_token"]

# Function to load the Stable Diffusion model from Hugging Face using the token
@st.cache_resource
def load_model():
    text2img_client = InferenceClient(
        "black-forest-labs/FLUX.1-dev", token=hf_token)
    return text2img_client

# Function to load the image-to-image model
@st.cache_resource
def load_img2img_model():
    img2img_client = InferenceClient(
        "stabilityai/stable-diffusion-2-1-unclip", token=hf_token)
    return img2img_client

# Loading the models once during the app initialization
text2img_client = load_model()
img2img_client = load_img2img_model()

# Title
st.markdown("<h1 align='center'>Sketch-image</h1><h4 align='center' style='font-weight: normal;'>Transform your imagination into images from text or sketches</h4>", unsafe_allow_html=True)


# Sidebar configuration: Allow users to customize image style, resolution, parameters, and select mode
with st.sidebar:
    # Option menu to switch between "Generate with Prompt" and "Generate with Image"
    option = st.radio(
        label="Choose Generation Mode",
        options=["Generate with Prompt", "Generate with Image"],
        index=0,
        help="Switch between text-to-image and image-to-image generation."
    )

    with st.expander("**Image Customization**", icon="🖌️", expanded=True):
        # Option to select the style of the generated image
        style = st.radio(label="Select Style", options=["Default", "Photorealistic", "Anime"], index=0, horizontal=True,
                         help="Choose the style of the image. 'Photorealistic' gives a more lifelike look, 'Anime' adds a stylized animated appearance.")

        # Option to select the resolution for the generated image
        image_resolution = st.radio("Image Resolution", options=["512x512", "768x768", "1024x1024"], index=0, horizontal=True,
                                    help="Select the resolution of the image. Higher resolutions offer more details but take longer to generate.")
        # Extract width and height for image resolution
        width, height = map(int, image_resolution.split('x'))

    with st.expander("**Parameter Customization**", icon="🛠️"):
        # Slider for adjusting inference steps (impact on image detail and generation time)
        inference_steps = st.slider("**Inference Steps**", min_value=10, max_value=100, value=50, step=1,
                                    help="Adjust the number of steps for generating the image. Higher values result in more detailed images but take longer to generate.")

        # Slider for adjusting the guidance scale (controls how closely the model follows the prompt)
        guidance_scale = st.slider("**Guidance Scale**", min_value=5.0, max_value=20.0, value=7.5, step=0.5,
                                   help="Control how closely the generated image aligns with your prompt. Higher values make the image more faithful to the prompt.")

        # For image-to-image transformation strength
        if option == "Generate with Image":
            strength = st.slider("**Transformation Strength**", min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                                help="Controls how much to transform the original image. Higher values mean stronger transformation.")


# ...existing code...
# Generate image using text prompt if "Generate with Prompt" is selected
if option == "Generate with Prompt":

    # Input field for the user to enter a text prompt
    prompt = st.text_area("Enter your prompt", key="prompt",
                          placeholder="A futuristic cityscape at sunset", height=70)

    # Layout for the button and image display
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col2:

        # Button to trigger the image generation process
        generate_button = st.button(
            "✨ Generate", use_container_width=True, type='primary')

    # Image generation logic when the button is clicked
    if generate_button:
        if prompt:
            _, col, _ = st.columns([1.5, 1, 1.5])
            with col:
                with st.spinner("Generating image..."):
                    try:
                        # Modify the prompt if the style is not 'Default'
                        if style != "Default":
                            prompt += f" in {style} style."

                        # Call the model's text_to_image function to generate the image
                        # LLM API CALL - Text to Image Generation
                        # This is where the prompt is sent to the model to generate an image
                        # You can replace this with a different API by changing the client and parameters
                        image = text2img_client.text_to_image(
                            prompt=prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=inference_steps,
                            width=width,
                            height=height
                        )

                        if image:

                            # Save the generated image in session state for persistence across interactions
                            st.session_state["generated_image"] = image

                            # Convert the image to a byte stream for download
                            image_bytes = BytesIO()
                            image.save(image_bytes, format="PNG")
                            image_bytes.seek(0)
                            image_from_bytes = Image.open(image_bytes)
                            image_size = image_from_bytes.size  # Get the image size for the filename

                            with col3:

                                # Adjust style name if it is 'Default'
                                if style == "Default":
                                    style = ""

                                # Create a download button with the generated image and its resolution
                                st.download_button(
                                    label="Download Image",
                                    data=image_bytes,
                                    file_name=f"sketchgen_{style}_generated_image_{image_size[0]}x{image_size[1]}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )

                    except Exception as e:
                        # Handle any errors that occur during image generation
                        st.error(
                            f"An error occurred while generating the image: {e}")
        else:
            # Prompt user to enter a valid prompt if none is provided
            st.error("Please enter a prompt!")

    # Display the generated image if it is stored in session state
    if "generated_image" in st.session_state:
        st.image(st.session_state["generated_image"],
                 caption="Generated Image", use_container_width=True)


else:
    # The "Generate with Image" feature implementation
    prompt = st.text_area("Enter your prompt (Optional)", 
                          placeholder="Transform this sketch into a detailed image", 
                          height=70)
    
    uploaded_image = st.file_uploader("Upload an Image (Sketch or Reference)", type=["jpeg", "jpg", "png"])

    if uploaded_image:
        st.sidebar.image(uploaded_image, caption="Uploaded Image")

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col2:
        generate_button = st.button("✨ Generate", use_container_width=True, type='primary')

    # Prepare the prompt based on user input
    if not prompt:
        prompt = "Transform this sketch into a detailed image"
    elif style != "Default":
        prompt += f" in {style} style."

    if generate_button:
        if uploaded_image:
            pil_image = Image.open(uploaded_image).convert("RGB")
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            _, col, _ = st.columns([1.5, 1, 1.5])
            with col:
                with st.spinner("Generating image..."):
                    try:
                        # LLM API CALL - Image to Image Transformation
                        # This is where the uploaded image and prompt are sent to the model
                        # You can replace this with a different API by changing the client and parameters
                        generated_image = img2img_client.image_to_image(
                            image=img_byte_arr,
                            prompt=prompt,
                            strength=0.8,
                            guidance_scale=guidance_scale,
                            num_inference_steps=inference_steps
                        )

                        if generated_image:
                            # Save the generated image in session state
                            st.session_state["sketch_generated_image"] = generated_image
                            
                            # Convert the image to a byte stream for download
                            image_bytes = BytesIO()
                            generated_image.save(image_bytes, format="PNG")
                            image_bytes.seek(0)
                            
                            with col3:
                                # Adjust style name if it is 'Default'
                                if style == "Default":
                                    style = ""
                                    
                                # Create a download button
                                st.download_button(
                                    label="Download Image",
                                    data=image_bytes,
                                    file_name=f"sketchgen_sketch_to_image_{style}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        else:
                            st.error("Image generation failed.")
                    except Exception as e:
                        st.error(f"An error occurred while generating the image: {e}")
        else:
            st.error("Please upload an image.")
    
    # Display the generated image from sketch if available
    if "sketch_generated_image" in st.session_state:
        st.image(st.session_state["sketch_generated_image"],
                caption="Generated Image from Sketch", use_container_width=True)
