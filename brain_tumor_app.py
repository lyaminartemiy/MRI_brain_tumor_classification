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
            "##### Документация модели PyTorch: [efficientnet-b1](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html).")
        
        architecture = Image.open('web/efficient-net/architecture.png')
        perfomance = Image.open('web/efficient-net/perfomance.png')
        
        st.image(architecture, caption='Network Architecture')
        st.image(perfomance, caption='Network Perfomance')
        
    elif show_data == 'Types of tumors':
        
        st.subheader('Примеры МРТ снимков')
        
        tab1, tab2, tab3 = st.tabs(["Глиома", "Менингиома", "Гипофиз"])
        
        image1 = Image.open('web/Glioma.png')
        image2 = Image.open('web/Meningioma.png')
        image3 = Image.open('web/Pituitary.png')
        
        with tab1:
            st.image(image1, caption='Пример МРТ снимка: ГЛИОМА')
            st.divider()
            st.write("""Глиома головного мозга — наиболее распространенная опухоль головного мозга, берущая свое начало из различных клеток глии. Клинические проявления глиомы зависят от ее расположения и могут включать головную боль, тошноту, вестибулярную атаксию, расстройство зрения, парезы и параличи, дизартрию, нарушения чувствительности, судорожные приступы и прочее""")
            st.write("""Глиома головного мозга диагностируется по результатам МРТ головного мозга и морфологического исследования опухолевых тканей. Вспомогательное значение имеет проведение Эхо-ЭГ, ЭЭГ, ангиографии сосудов головного мозга, ЭЭГ, офтальмоскопии, исследования цереброспинальной жидкости, ПЭТ и сцинтиграфии. Общепринятыми способами лечения в отношении глиомы головного мозга являются хирургическое удаление, лучевая терапия, стереотаксическая радиохирургия и химиотерапия.""")
        
        with tab2:
            st.image(image2, caption='Пример МРТ снимка: МЕНИНГИОМА')
            st.divider()
            st.write("""Менингиома представляет собой опухоль, чаще всего доброкачественной природы, произрастающую из арахноэндотелия мозговых оболочек. Обычно опухоль локализуется на поверхности мозга (реже на конвекситальной поверхности либо на основании черепа, редко в желудочках, или в костной ткани). Как и для многих других доброкачественных опухолей, для менингиом характерен медленный рост. Довольно часто не дает о себе знать, вплоть до значительного увеличения новообразования; иногда бывает случайной находкой при компьютерной или магнитно-резонансной томографии.""")
            st.write("""В клинической неврологии менингиома по частоте встречаемости занимает второе место после глиом. Всего менингиомы составляют примерно 20-25% от всех опухолей центральной нервной системы. Менингиомы возникают преимущественно у людей в возрасте 35-70 лет; чаще всего наблюдаются у женщин. У детей встречаются довольно редко и составляют примерно 1,5% от всех детских новообразований ЦНС. 8-10% опухолей паутинной мозговой оболочки представлены атипичными и злокачественными менингиомами.""")
            
        with tab3:
            st.image(image3, caption='Пример МРТ снимка: ОПУХОЛЬ ГИПОФИЗА')
            st.divider()
            st.write("""Опухоли гипофиза – группа доброкачественных, реже – злокачественных новообразований передней доли (аденогипофиза) или задней доли (нейрогипофиза) железы. Опухоли гипофиза, по статистике, составляют около 15% новообразований внутричерепной локализации. Они одинаково часто диагностируются у лиц обоих полов, обычно в возрасте 30-40 лет. Подавляющее большинство опухолей гипофиза составляют аденомы, которые подразделяются на несколько видов в зависимости от размеров и гормональной активности. Симптомы опухоли гипофиза представляют собой сочетание признаков объемного внутримозгового процесса и гормональных нарушений. Диагностика опухоли гипофиза осуществляется проведением целого ряда клинических и гормональных исследований, ангиографии и МРТ головного мозга.""")
            st.write("""Опухоли гипофиза – группа доброкачественных, реже – злокачественных новообразований передней доли (аденогипофиза) или задней доли (нейрогипофиза) железы. Опухоли гипофиза, по статистике, составляют около 15% новообразований внутричерепной локализации. Они одинаково часто диагностируются у лиц обоих полов, обычно в возрасте 30-40 лет.""")
        
    elif show_data == 'Classify the tumor':
        
        with st.spinner('Нейросеть загружается...'):
            model = EfficientNetModel('efficientnet-b1', path_to_model_dict='my-efficientnet-b1')

        st.success('Нейросеть загружена! Всё готово к загрузке снимка!', icon="✅")
        
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
                st.write("Можно загрузить только файлы формата png или jpeg (jpg).")
                st.write("Попробуйте снова!")
                return 0
            
            classes = ['МЕНИНГИОМА', 'ГЛИОМА', 'ОПУХОЛЬ ГИПОФИЗА']
            
            st.image(image, caption='ЗАГРУЖЕННЫЙ СНИМОК')
            
            transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transforms(image)
            img_tensor = img_tensor.unsqueeze(0)
            
            batched_images = torch.cat([img_tensor, img_tensor, img_tensor, img_tensor], dim=0)
            
            probs = model.predict_item(batched_images)[0]
            
            with st.spinner('Подождите, снимок обрабатывается...'):
                sleep(3)
            
            index = np.argmax(probs)
            st.info('Результат классификации модели: ' + classes[index], icon="❗")
            st.info('Вероятность: ' + str(probs[index]), icon="❗")

        
if __name__ == "__main__":
    upload()
