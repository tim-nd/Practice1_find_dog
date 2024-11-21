from fastai.vision.all import *
import gradio as gr

def is_dog(x): return x[0].islower()

learn = load_learner('model.pkl')

categories = ('Cat', 'Dog')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image(height = 192, width = 192)
label = gr.Label()
examples = ['dog.jpeg', 'cat.jpeg']

intf = gr.Interface(fn = classify_image, inputs = image, outputs = label, examples = examples)
intf.launch(inline = False)