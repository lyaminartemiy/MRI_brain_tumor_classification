# 🧠 MRI brain tumor classifier

### What is it?
> Web interface for classification of brain tumors on MRI images. <br>
> Link to the web interface: https://mribraintumorclassifier.streamlit.app

### Content
> The data is the collection of MRI Images of 3 types of Brain Tumor, Pituitary, Meningioma and Glioma Tumor in GrayScale Format.

### Examples of MRI images with tumors:
![image](https://i.ibb.co/LSwCS1B/Glioma.png) <br>
*Example of an MRI image: GLIOMA*

Brain glioma is the most common brain tumor originating from various glial cells. Clinical manifestations of glioma depend on its location and may include headache, nausea, vestibular ataxia, visual impairment, paresis and paralysis, dysarthria, sensitivity disorders, seizures, etc.

---

![image](https://i.ibb.co/QHH7xVN/Meningioma.png) <br>
*Example of an MRI image: MENINGIOMA*

Meningioma is a tumor, most often of a benign nature, growing from the arachnoendothelium of the meninges. Usually the tumor is localized on the surface of the brain (less often on the convexital surface or on the base of the skull, rarely in the ventricles, or in bone tissue). As with many other benign tumors, meningiomas are characterized by slow growth. Quite often it does not make itself felt, up to a significant increase in the neoplasm; sometimes it happens to be an accidental finding with computer or magnetic resonance imaging.

---

![image](https://i.ibb.co/CbvBTNv/Pituitary.png) <br>
*Example of an MRI image: PITUITARY TUMOR*

Pituitary tumors are a group of benign, rarely malignant neoplasms of the anterior lobe (adenohypophysis) or posterior lobe (neurohypophysis) of the gland. Pituitary tumors, according to statistics, account for about 15% of tumors of intracranial localization. They are equally often diagnosed in persons of both sexes, usually at the age of 30-40 years. The vast majority of pituitary tumors are adenomas, which are divided into several types depending on the size and hormonal activity. The symptoms of a pituitary tumor are a combination of signs of a volumetric intracerebral process and hormonal disorders. Diagnosis of a pituitary tumor is carried out by a number of clinical and hormonal studies, angiography and MRI of the brain.

---

### Solution

**Transform Data**:
- The `BrainTumorDataset` class is written for storing and transforming source data. <br>
- Accepts `transform` (`str` or `torchvision.transforms.Transform`): transformation applied to images.
- Training: resize to `3 x 224 x 224`, random rotation in degrees `-45:45`, random horizontal flip, random vertical flip, transformation to tensor, normalizes with mean `0.485, 0.456, 0.406` and standard deviation `0.229, 0.224, 0.225`.
- Testing: resize to `3 x 224 x 224`, transformation to tensor, normalizes with mean `0.485, 0.456, 0.406` <br> and standard deviation `0.229, 0.224, 0.225`.

**Network Training**:
- The training is presented in the file `model_train.ipynb`.
- Efficient-net-b1 is used as a prepaid model, the output of the neural network is changed by 3 neurons (the number of our classes), and the first 3 layers are frozen to prevent retraining on our data.
- After receiving the information, all features are saved using `torch.save(model.state_dict(), path)` in the `my-efficient-net-b1` file.

![image](https://i.ibb.co/bHLNBHV/2023-05-29-125651584.png) <br>
*Graph of the loss function during network training and testing*

![image](https://i.ibb.co/RPxws4B/2023-05-29-125852201.png) <br>
*Graph of accuracy during network training and testing*

**Web Interface**:
- The web interface is presented in the file `brain_tumor_app.py`.
- The web interface is made using the `Streamlit` framework.
- Also, all content is located in the `web` directory.
