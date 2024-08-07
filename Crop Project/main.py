import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Function to apply CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the CSS
local_css("style.css")
#tensorflow model prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model("NEW_CROP_DISEASE_DETECTION_MODEL.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index

#sidebar
# logo_path = "dash1.webp"

# # Create the HTML string to display the image and text in the same line
# logo_html = f'''
# '''

# # Use st.sidebar.markdown to render the HTML
# st.sidebar.markdown(logo_html, unsafe_allow_html=True)
st.sidebar.write("""
                 <div style="display: flex; align-items: center;">
                 <span style="font-size: 24px; margin-left: 10px;">🍀AgriScan</span>
                 </div>
                 """,unsafe_allow_html=True)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


#Main Page
if(app_mode=="Home"):
    st.header("Welcome to AgriScan🍀!")
    st.subheader("CROP DISEASE RECOGNITION SYSTEM")
    image_path = "cover.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Crop Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying crop diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Crop Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                ### Project Description
                The Crop Disease Recognition System aims to provide a solution for identifying diseases in crop leaves efficiently. By leveraging machine learning techniques, the system analyzes images of healthy and diseased crop leaves to detect signs of diseases, ultimately aiding in the early detection and management of crop diseases.

                ### Dataset Information
                The dataset used in this project is a curated collection of RGB images of healthy and diseased crop leaves. It comprises approximately 87,000 images categorized into 38 different classes, representing various types of crop diseases. The dataset is created using offline augmentation techniques from the original dataset, which can be found on [GitHub](link-to-github-repo). 

                ### Content Overview
                The dataset is divided into three main subsets:
                1. **Training Set:** Consisting of 70,295 images, the training set is used to train the machine learning model. 
                2. **Validation Set:** With 17,572 images, the validation set is utilized to fine-tune model parameters and prevent overfitting.
                3. **Test Set:** Comprising 33 images, the test set is reserved for evaluating the model's performance and making predictions.

                ### Future Work
                Future plans for the project may include expanding the dataset, improving the model architecture, and enhancing the user interface for a more seamless experience. Additionally, incorporating real-time disease detection and extending the system to cover a wider range of crop types are potential areas for further development.

                ### Acknowledgements
                We acknowledge the original dataset creators and contributors for their valuable resources. Their efforts have been instrumental in advancing research and development in the field of crop disease recognition.
                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")

    st.markdown("### Supported Diseases")

    dataset_description = {
        'Name': ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato'],
        'No of Classes': [4, 1, 2, 4, 4, 1, 2, 2, 3, 1, 1, 1, 2, 10],
        'Class Names': [
            "'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy'",
            "'Blueberry___healthy'",
            "'Cherry_(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy'",
            "'Corn__Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn__healthy'",
            "'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'",
            "'Orange__Haunglongbing(Citrus_greening)'",
            "'Peach__Bacterial_spot', 'Peach__healthy'",
            "'Pepper,bell_Bacterial_spot', 'Pepper,_bell__healthy'",
            "'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy'",
            "'Raspberry___healthy'",
            "'Soybean___healthy'",
            "'Squash___Powdery_mildew'",
            "'Strawberry__Leaf_scorch', 'Strawberry__healthy'",
            "'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'"
        ]
    }
    st.table(pd.DataFrame(dataset_description))
    
    test_image = st.file_uploader("Share a picture of the plant and get immediate results!")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        # st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

image_path = "dash1.webp"
st.sidebar.image(image_path, width=300)