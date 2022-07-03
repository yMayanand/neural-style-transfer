import cv2
import numpy as np
from torchvision import transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def gram_matrix(x):
    """computes gram matrix of feature map shape (b, ch, h, w)"""
    b, ch, h, w = x.shape
    features = x.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def load_image(img_path, shape=None):
    """load image given its path"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if shape is not None:
        img = cv2.resize(img, shape)
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img

def preprocess_image(image):
    """preprocesses image and prepares it to be fed to the model"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    img = transform(image)
    img = img.unsqueeze(0)
    return img

def save_image(filename, data):
    """saves image after training"""
    img = postprocess_image(data)
    img = img * 255
    img = img.astype("uint8")
    cv2.imwrite(filename, img[:, :, ::-1]) # converts rgb to bgr due to opencv constraint

def postprocess_image(image):
    """postprocesses images after training"""
    image = image.squeeze(0)
    image = image.cpu().detach().clone().clamp(0, 1).numpy()
    image = image.transpose(1, 2, 0)
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    return image



    


