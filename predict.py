
import gradio as gr
from model import load_model, predict_image
from PIL import Image

model = load_model('trained_net.pth')

# Gradio interface function
def gradio_predict(image):
    return predict_image(image, model)

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="CIPHAR 10 dataset classification",
    description="Upload an image to get the predicted class using the trained model." 
)

if __name__ == "__main__":
    iface.launch()
