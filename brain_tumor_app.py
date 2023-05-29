import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

from efficientnet_pytorch import EfficientNet

import numpy as np
from PIL import Image
from tqdm import trange
from time import sleep
from matplotlib import pyplot as plt

import streamlit as st

import sys
import zipfile
from EfficientNetModel import EfficientNetModel


st.set_page_config(
    page_title="Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("BRAIN TUMOR CLASSIFIER")
st.write("""Web interface for downloading MRI images and classifying brain tumors of 3 types: glioma, meningioma and pituitary tumor.""")

# Настройка боковой панели
st.sidebar.title("What is a brain tumor?")
st.sidebar.info(
    """
    A brain tumor is a cluster or mass of abnormal cells in your brain. Your skull, which surrounds your brain, is very hard. Any growth inside such a limited space can cause problems.
    """
)
st.sidebar.info(
    """
    Brain tumors can be cancerous (malignant) or non-cancerous (benign). When benign or malignant tumors grow, they can lead to increased pressure inside your skull. This can cause brain damage and be life-threatening.
    """
)

# Основная функция веб-интерфейса
def upload():
    
    st.divider()
    
    show_data = st.sidebar.selectbox('Select an action', ('Classify the tumor', 'Types of tumors', 'Neural network'))
    
    if show_data == 'Neural network':
        
        st.subheader('Архитектура модели')
        st.markdown(
            "##### Documentation of the PyTorch model: [efficientnet-b1](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html).")
        
        architecture = Image.open('web/efficient-net/architecture.png')
        perfomance = Image.open('web/efficient-net/perfomance.png')
        
        st.image(architecture, caption='Network Architecture')
        st.image(perfomance, caption='Network Perfomance')
        
    elif show_data == 'Types of tumors':
        
        st.subheader('Examples of MRI images')
        
        tab1, tab2, tab3 = st.tabs(["Glioma", "Meningioma", "Pituitary Tumor"])
        
        image1 = Image.open('web/Glioma.png')
        image2 = Image.open('web/Meningioma.png')
        image3 = Image.open('web/Pituitary.png')
        
        with tab1:
            st.image(image1, caption='Example of an MRI image: GLIOMA')
            st.divider()
            st.write("""Brain glioma is the most common brain tumor originating from various glial cells. Clinical manifestations of glioma depend on its location and may include headache, nausea, vestibular ataxia, visual impairment, paresis and paralysis, dysarthria, sensitivity disorders, seizures, etc.""")
            st.write("""Glioma of the brain is diagnosed according to the results of MRI of the brain and morphological examination of tumor tissues. It is of auxiliary importance to carry out Echo-EG, EEG, angiography of cerebral vessels, EEG, ophthalmoscopy, examination of cerebrospinal fluid, PET and scintigraphy. Conventional methods of treatment for brain glioma are surgical removal, radiation therapy, stereotactic radiosurgery and chemotherapy.""")
        
        with tab2:
            st.image(image2, caption='Example of an MRI image: MENINGIOMA')
            st.divider()
            st.write("""Meningioma is a tumor, most often of a benign nature, growing from the arachnoendothelium of the meninges. Usually the tumor is localized on the surface of the brain (less often on the convexital surface or on the base of the skull, rarely in the ventricles, or in bone tissue). As with many other benign tumors, meningiomas are characterized by slow growth. Quite often it does not make itself felt, up to a significant increase in the neoplasm; sometimes it happens to be an accidental finding with computer or magnetic resonance imaging.""")
            st.write("""In clinical neurology, meningioma ranks second after gliomas in frequency of occurrence. In total, meningiomas account for about 20-25% of all tumors of the central nervous system. Meningiomas occur mainly in people aged 35-70 years; they are most often observed in women. In children, they are quite rare and account for about 1.5% of all childhood neoplasms of the central nervous system. 8-10% of tumors of the arachnoid meningioma are represented by atypical and malignant meningiomas.""")
            
        with tab3:
            st.image(image3, caption='Example of an MRI image: PITUITARY TUMOR')
            st.divider()
            st.write("""Pituitary tumors are a group of benign, rarely malignant neoplasms of the anterior lobe (adenohypophysis) or posterior lobe (neurohypophysis) of the gland. Pituitary tumors, according to statistics, account for about 15% of tumors of intracranial localization. They are equally often diagnosed in persons of both sexes, usually at the age of 30-40 years. The vast majority of pituitary tumors are adenomas, which are divided into several types depending on the size and hormonal activity. The symptoms of a pituitary tumor are a combination of signs of a volumetric intracerebral process and hormonal disorders. Diagnosis of a pituitary tumor is carried out by a number of clinical and hormonal studies, angiography and MRI of the brain.""")
            st.write("""Pituitary tumors are a group of benign, rarely malignant neoplasms of the anterior lobe (adenohypophysis) or posterior lobe (neurohypophysis) of the gland. Pituitary tumors, according to statistics, account for about 15% of tumors of intracranial localization. They are equally often diagnosed in persons of both sexes, usually at the age of 30-40 years.""")
        
    elif show_data == 'Classify the tumor':
        
        with st.spinner('The neural network is being loaded...'):
            model = EfficientNetModel('efficientnet-b1', path_to_model_dict='my-efficientnet-b1')

        st.success('The neural network is loaded! Everything is ready to upload the MRI image!', icon="✅")
        
        # Загрузка файла
        uploaded_file = st.file_uploader("Выберите файл")

        # Проверка на допустимость формата файла
        if uploaded_file is not None:

            # Извлечение формата
            format_name = uploaded_file.name.split('.')[1]

            # Проверка формата
            if format_name == "png":
                image = Image.open(uploaded_file)
            elif format_name == "jpeg":
                image = Image.open(uploaded_file)
            elif format_name == "jpg":
                image = Image.open(uploaded_file)
            else:
                st.write("You can only upload png or jpeg (jpg) files.")
                st.write("Try again!")
                return 0
            
            classes = ['MENINGIOMA', 'GLIOMA', 'PITUITARY TUMOR']
            
            st.image(image, caption='UPLOADED MRI IMAGE')
            
            transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transforms(image)
            img_tensor = img_tensor.unsqueeze(0)
            
            batched_images = torch.cat([img_tensor, img_tensor, img_tensor, img_tensor], dim=0)
            
            probs = model.predict_item(batched_images)[0]
            
            with st.spinner('Wait, the snapshot is being processed...'):
                sleep(3)
            
            index = np.argmax(probs)
            st.info('Model classification result: ' + classes[index], icon="❗")
            st.info('Probability: ' + str(probs[index]), icon="❗")

        
if __name__ == "__main__":
    upload()
