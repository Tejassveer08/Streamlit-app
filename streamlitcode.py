import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval();
    return model

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()

    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1,2,0)

    return feature_imp

st.title("ResNet-50 Image Classifier :tea: :coffee:") #names of emojis
upload = st.file_uploader(label="Upload Image :", type=["png", "jpg", "jpeg"]) #label for the file uploader

if upload: #uploaded image will be available using this variable
    img = Image.open(upload) #convert bytestring image to pillow image

    processed_img = preprocess_func(img) #preprocess the image using the ResNet50 model's preprocess function
    model = load_model() #load the model
    probs, idxs = make_prediction(model, processed_img) #make prediction using the model and the preprocessed image
    feature_imp = interpret_prediction(model, processed_img, idxs[0]) #interpret the prediction using the integrated gradients algorithm

    interp_fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6,6)) #visualize the feature importance using matplotlib     
    prob_fig = plt.figure(figsize=(12,2.5)) #create a new figure for displaying the top 5 probabilities
    ax = prob_fig.add_subplot(111) #create a new figure and axis object
    plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4+["tomato"]) #bar chart of top 5 probabilities
    plt.title('Top 5 Probabilities', loc="center", fontsize=15) #title, location and font size of the bar chart
    st.pyplot(prob_fig, use_container_width=True) #display the bar chart

    col1, col2 = st.columns(2, gap="medium") #create two columns with a medium gap between them

    with col1:
        main_fig = plt.figure(figsize=(6,6)) #create a new figure for displaying the feature importance
        ax = main_fig.add_subplot(111) 
        plt.imshow(img);
        plt.xticks([],[]); 
        plt.yticks([],[]);
        st.pyplot(main_fig, use_container_width=True)

    with col2:
        st.pyplot(interp_fig, use_container_width=True) #display the feature importance visualization