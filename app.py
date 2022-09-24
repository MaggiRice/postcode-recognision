#libraries to set up
from flask import Flask, render_template, request
from keras.models import load_model

#libraries for image processing
import cv2
from cgitb import grey
import numpy as np 
import os

app = Flask(__name__, template_folder='./template')

cnn_model = load_model('./models/cnn.h5')

#functions to process the image
def sobel(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def edge_detect(img):
    return np.max(np.array([sobel(img[:,:, 0]), sobel(img[:,:, 1]), sobel(img[:,:, 2]) ]), axis=0)

def scan_save(img_path):
    # Image pre-processing - blur, edges, threshold, closing
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(image, (5,5), 18)
    edges = edge_detect(blurred)
    ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    bw_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
    
    #after scan - save the entire img
    path = './static/scanned'
    cv2.imwrite(os.path.join(path , 'scan1.jpg'), bw_image)
    
    #after process - save each digit
    edges,dilated = process(bw_image)
    new_image,contours,res = manage_contours(edges,bw_image.copy())
    res = process_individual_images(res)
    return res   #this is imoprtant


#functions to extract each digits
def process(image):
    ret,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    struct = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh ,struct,iterations=1)
    edges = cv2.Canny(dilated,500,5)
    return edges,dilated

def manage_contours(image,orig_image):
    results=[]
    contours,hier = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours,key=cv2.boundingRect)
    cv2.drawContours(orig_image,contours,-1,(255,255,255),2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        small_image = orig_image[y:y+h,x:x+w]
        results.append(small_image)
    return orig_image,contours,results

def process_individual_images(ilist):
    dil = []
    for img in (ilist):
        edge, dilated = process(img)
        dil.append(dilated)
    return dil
        
    

# def get_processed_images(ilist):
#     res = []
#     for img in ilist:
#         edg,dil = process(img)
#         res.append((edg,dil))
#     return res

def delete_all_files():
    directory1 = './static/extracted'  # folder with images
    for filename in os.scandir(directory1):
        if filename.is_file():
            os.remove(filename.path)
    directory2 = './static/scanned'  # folder with images
    for filename in os.scandir(directory2):
        if filename.is_file():
            os.remove(filename.path)
    directory3 = './static/uploaded'  # folder with images
    for filename in os.scandir(directory3):
        if filename.is_file():
            os.remove(filename.path)

#functions to predict the labels
directory = './static/extracted'
def cnn_predict(small_images):
    predicted_cnn = []
    for image in small_images:
        img = image
        img = cv2.resize(img, (28,28))
        # img = img[:, :, 0]
        img = img.reshape(1,28,28,1)
        img = img.astype('float32')
        img = img/255.0
        ans = np.argmax(cnn_model.predict(img))
        predicted_cnn.append(str(ans))
    return predicted_cnn    

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    #clear all images within folder when start the website
    delete_all_files()
    #if there if a post request
    if request.method == 'POST':
        result = []
        ori_img = request.files['img_uploaded']
        img_path = "./static/uploaded/ori_" + ori_img.filename
        ori_img.save(img_path)
        result = scan_save(img_path)

        cnn_pred= []
        cnn_pred = cnn_predict(result)
        detected_ans = ""
        for ans in cnn_pred:
            detected_ans += ans
            
        return render_template("./index.html", prediction = detected_ans, img_path = img_path)
    
    #else just render the page
    return render_template("./index.html")

if __name__ =='__main__':
	app.run(debug = True)