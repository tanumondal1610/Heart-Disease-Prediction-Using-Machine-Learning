import numpy as np
import pickle
from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.signal
from keras.models import Sequential
from werkzeug.utils import secure_filename
from keras.layers import LSTM, Dense, Reshape
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
import plotly.graph_objs as go
import plotly.express as px
from scipy.signal import medfilt, butter, filtfilt
import pywt
from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted

# Load ML model
model = pickle.load(open('heartmodel.pkl', 'rb')) 
MODEL_PATH = 'ecg.pkl'

# Create application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')
    
@app.route('/image', methods=['GET'])
def image():
    # Main page
    return render_template('image.html')    

@app.route('/ecg')
def ecg():
    return render_template('ecg.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is not likely to have heart disease!')

@app.route('/predictsensor', methods =['POST'])
def predictsensor():
    features = [float(i) for i in request.form.values()]
    df = pd.read_csv('ecg.csv', header=None)
    abnormal = df[df.loc[:,140] ==0][:10]
    normal = df[df.loc[:,140] ==1][:10]
    # Create the figure 
    fig = go.Figure()
    #create a list to display only a single legend 
    leg  = [False] * abnormal.shape[0]
    leg[0] = True
    # split the data into labels and features 
    ecg_data = df.iloc[:,:-1]
    labels = df.iloc[:,-1]

    # Normalize the data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ecg_data = scaler.fit_transform(ecg_data)
    # Median filtering
    ecg_medfilt = medfilt(ecg_data, kernel_size=3)

    # Low-pass filtering
    lowcut = 0.05
    highcut = 20.0
    nyquist = 0.5 * 360.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    ecg_lowpass = filtfilt(b, a, ecg_data)

    # Wavelet filtering
    coeffs = pywt.wavedec(ecg_data, 'db4', level=1)
    threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(ecg_data)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    ecg_wavelet = pywt.waverec(coeffs, 'db4')
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(ecg_wavelet, labels, test_size=0.2, random_state=42)
    # Initializing an empty list to store the features
    
    features = []

    # Extracting features for each sample
    for i in range(X_train.shape[0]):
        #Finding the R-peaks
        r_peaks = scipy.signal.find_peaks(X_train[i])[0]

        #Initialize lists to hold R-peak and T-peak amplitudes
        r_amplitudes = []
        t_amplitudes = []

        # Iterate through R-peak locations to find corresponding T-peak amplitudes
        for r_peak in r_peaks:
            # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
            t_peak = np.argmin(X_train[i][r_peak:r_peak+200]) + r_peak
            #Append the R-peak amplitude and T-peak amplitude to the lists
            r_amplitudes.append(X_train[i][r_peak])
            t_amplitudes.append(X_train[i][t_peak])

        # extracting singular value metrics from the r_amplitudes
        std_r_amp = np.std(r_amplitudes)
        mean_r_amp = np.mean(r_amplitudes)
        median_r_amp = np.median(r_amplitudes)
        sum_r_amp = np.sum(r_amplitudes)
        # extracting singular value metrics from the t_amplitudes
        std_t_amp = np.std(t_amplitudes)
        mean_t_amp = np.mean(t_amplitudes)
        median_t_amp = np.median(t_amplitudes)
        sum_t_amp = np.sum(t_amplitudes)

        # Find the time between consecutive R-peaks
        rr_intervals = np.diff(r_peaks)

        # Calculate the time duration of the data collection
        time_duration = (len(X_train[i]) - 1) / 1000 # assuming data is in ms

        # Calculate the sampling rate
        sampling_rate = len(X_train[i]) / time_duration

        # Calculate heart rate
        duration = len(X_train[i]) / sampling_rate
        heart_rate = (len(r_peaks) / duration) * 60

        # QRS duration
        qrs_duration = []
        for j in range(len(r_peaks)):
            qrs_duration.append(r_peaks[j]-r_peaks[j-1])
        # extracting singular value metrics from the qrs_durations
        std_qrs = np.std(qrs_duration)
        mean_qrs = np.mean(qrs_duration)
        median_qrs = np.median(qrs_duration)
        sum_qrs = np.sum(qrs_duration)

        # Extracting the singular value metrics from the RR-interval
        std_rr = np.std(rr_intervals)
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)
        sum_rr = np.sum(rr_intervals)

        # Extracting the overall standard deviation 
        std = np.std(X_train[i])
        
        # Extracting the overall mean 
        mean = np.mean(X_train[i])

        # Appending the features to the list
        features.append([mean, std, std_qrs, mean_qrs,median_qrs,std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp,heart_rate])
        
    # Converting the list to a numpy array
    
    features = np.array(features)
        # Initializing an empty list to store the features
    X_test_fe = []

    # Extracting features for each sample
    for i in range(X_test.shape[0]):
        # Finding the R-peaks
        r_peaks = scipy.signal.find_peaks(X_test[i])[0]

        # Initialize lists to hold R-peak and T-peak amplitudes
        r_amplitudes = []
        t_amplitudes = []

        # Iterate through R-peak locations to find corresponding T-peak amplitudes
        for r_peak in r_peaks:
            # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
            t_peak = np.argmin(X_test[i][r_peak:r_peak+200]) + r_peak
            # Append the R-peak amplitude and T-peak amplitude to the lists
            r_amplitudes.append(X_test[i][r_peak])
            t_amplitudes.append(X_test[i][t_peak])
        #extracting singular value metrics from the r_amplitudes
        std_r_amp = np.std(r_amplitudes)
        mean_r_amp = np.mean(r_amplitudes)
        median_r_amp = np.median(r_amplitudes)
        sum_r_amp = np.sum(r_amplitudes)
        #extracting singular value metrics from the t_amplitudes
        std_t_amp = np.std(t_amplitudes)
        mean_t_amp = np.mean(t_amplitudes)
        median_t_amp = np.median(t_amplitudes)
        sum_t_amp = np.sum(t_amplitudes)

        # Find the time between consecutive R-peaks
        rr_intervals = np.diff(r_peaks)

        # Calculate the time duration of the data collection
        time_duration = (len(X_test[i]) - 1) / 1000 # assuming data is in ms

        # Calculate the sampling rate
        sampling_rate = len(X_test[i]) / time_duration

        # Calculate heart rate
        duration = len(X_test[i]) / sampling_rate
        heart_rate = (len(r_peaks) / duration) * 60

        # QRS duration
        qrs_duration = []
        for j in range(len(r_peaks)):
            qrs_duration.append(r_peaks[j]-r_peaks[j-1])
        #extracting singular value metrics from the qrs_duartions
        std_qrs = np.std(qrs_duration)
        mean_qrs = np.mean(qrs_duration)
        median_qrs = np.median(qrs_duration)
        sum_qrs = np.sum(qrs_duration)

        # Extracting the standard deviation of the RR-interval
        std_rr = np.std(rr_intervals)
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)
        sum_rr = np.sum(rr_intervals)
        
          # Extracting the standard deviation of the RR-interval
        std = np.std(X_test[i])
        
        # Extracting the mean of the RR-interval
        mean = np.mean(X_test[i])

        # Appending the features to the list
        X_test_fe.append([mean, std, std_qrs, mean_qrs,median_qrs,std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp,heart_rate])
        
    
#     df1 = pd.DataFrame(X_test_fe)
#     df2= pd.DataFrame(y_test)
#     df1.reset_index(drop=True, inplace=True)
#     df2.reset_index(drop=True, inplace=True)
#     frames = [df1, df2]
#     
#     result = pd.concat(frames,axis=1, ignore_index=True)
#     result.to_csv('out.csv')
#     print(result.head())
    
    # Converting the list to a numpy array
    X_test_fe = np.array(X_test_fe)
    
        # Define the number of features in the train dataframe
    num_features = features.shape[1]

    # Reshape the features data to be in the right shape for LSTM input
    features = np.asarray(features).astype('float32')
    features = features.reshape(features.shape[0], features.shape[1], 1)

    X_test_fe = np.asarray(X_test_fe).astype('float32')
    X_test_fe = X_test_fe.reshape(X_test_fe.shape[0], X_test_fe.shape[1], 1)

    # Define the model architecture
    model = Sequential()
    model.add(LSTM(64, input_shape=(features.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(features, y_train, validation_data=(X_test_fe, y_test), epochs=4, batch_size=32)
    my_array = np.asarray(features)
    l=my_array.reshape(1,51974)
    l=l.astype('float32')
    
    #print("Type=",type(l))
    #print("Shape=",l.shape)
    #print("Data type=",l.dtype)
    # Make predictions on the validation set
    pred = model.predict(l)
    print("pred value=",pred)
    dataframe = str(df.head())
        # Check the output values and retrive the result with html tag based on the value
    if pred >= 0.5:
        return render_template('ECG.html', 
                               result = 'The patient is likely to have heart disease')
    else:
        return render_template('ECG.html', 
                               result = 'The patient is not likely to have heart disease')


@app.route('/predictimage', methods=['GET', 'POST'])
def upload():
    predicted_result=''
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        uploaded_file = st.file_uploader(file_path)
        image=imread("History_Of_MI/PMI(1).jpg")
        image_gray = color.rgb2gray(image)
        image_gray=resize(image_gray,(1572,2213))

        image1=imread('History_Of_MI/PMI(1).jpg')
        image1 = color.rgb2gray(image1)
        image1=resize(image1,(1572,2213))  

        image2=imread('AbnormalHeartbeat/HB(6).jpg')
        image2 = color.rgb2gray(image2)
        image2=resize(image2,(1572,2213))

        image3=imread('Normal/Normal(1).jpg')
        image3 = color.rgb2gray(image3)
        image3=resize(image2,(1572,2213))

        image4=imread('MyocardialInfarction/MI(1).jpg')
        image4 = color.rgb2gray(image4)
        image4=resize(image2,(1572,2213))

        #similarity_score=max(structural_similarity(image_gray,image1,data_range=None,),
        #                     structural_similarity(image_gray,image2,data_range=None,),
        #                     structural_similarity(image_gray,image3,data_range=None,),
        #                     structural_similarity(image_gray,image4,data_range=None,))

        #if similarity_score > 0.70:
            #st.image(image)

        #    my_expander = st.expander(label='Gray SCALE IMAGE')
        #    with my_expander: 
                #st.image(image_gray)
              

        Lead_1 = image[300:600, 150:643]
        Lead_2 = image[300:600, 646:1135]
        Lead_3 = image[300:600, 1140:1625]
        Lead_4 = image[300:600, 1630:2125]
        Lead_5 = image[600:900, 150:643]
        Lead_6 = image[600:900, 646:1135]
        Lead_7 = image[600:900, 1140:1625]
        Lead_8 = image[600:900, 1630:2125]
        Lead_9 = image[900:1200, 150:643]
        Lead_10 = image[900:1200, 646:1135]
        Lead_11 = image[900:1200, 1140:1625]
        Lead_12 = image[900:1200, 1630:2125]
        Lead_13 = image[1250:1480, 150:2125]
        Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]

        fig , ax = plt.subplots(4,3)
        fig.set_size_inches(10, 10)
        x_counter=0
        y_counter=0
        print("Hello")
        for x,y in enumerate(Leads[:len(Leads)-1]):
            if (x+1)%3==0:
                ax[x_counter][y_counter].imshow(y)
                ax[x_counter][y_counter].axis('off')
                ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
                x_counter+=1
                y_counter=0
            else:
                ax[x_counter][y_counter].imshow(y)
                ax[x_counter][y_counter].axis('off')
                ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
                y_counter+=1
    
        fig.savefig('Leads_1-12_figure.png')
        fig1 , ax1 = plt.subplots()
        fig1.set_size_inches(10, 10)
        ax1.imshow(Lead_13)
        ax1.set_title("Leads 13")
        ax1.axis('off')
        fig1.savefig('Long_Lead_13_figure.png')
        my_expander1 = st.expander(label='DIVIDING LEAD')
        with my_expander1:
            #st.image('Leads_1-12_figure.png')
            #st.image('Long_Lead_13_figure.png')
            pass

        """#### **PREPROCESSED LEADS**"""
        fig2 , ax2 = plt.subplots(4,3)
        fig2.set_size_inches(10, 10)
        #setting counter for plotting based on value
        x_counter=0
        y_counter=0

        for x,y in enumerate(Leads[:len(Leads)-1]):
            #converting to gray scale
            grayscale = color.rgb2gray(y)
            #smoothing image
            blurred_image = gaussian(grayscale, sigma=0.9)
            #thresholding to distinguish foreground and background
            #using otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            #creating binary image based on threshold
            binary_global = blurred_image < global_thresh
            #resize image
            binary_global = resize(binary_global, (300, 450))
            if (x+1)%3==0:
                ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
                ax2[x_counter][y_counter].axis('off')
                ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
                x_counter+=1
                y_counter=0
            else:
                ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
                ax2[x_counter][y_counter].axis('off')
                ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
                y_counter+=1
        fig2.savefig('Preprossed_Leads_1-12_figure.png')
    
        #plotting lead 13
        fig3 , ax3 = plt.subplots()
        fig3.set_size_inches(10, 10)
        #converting to gray scale
        grayscale = color.rgb2gray(Lead_13)
        #smoothing image
        blurred_image = gaussian(grayscale, sigma=0.7)
        global_thresh = threshold_otsu(blurred_image)
        print(global_thresh)
        #creating binary image based on threshold
        binary_global = blurred_image < global_thresh
        ax3.imshow(binary_global,cmap='gray')
        ax3.set_title("Leads 13")
        ax3.axis('off')
        fig3.savefig('Preprossed_Leads_13_figure.png')

        my_expander2 = st.expander(label='PREPROCESSED LEAD')
        with my_expander2:
            st.image('Preprossed_Leads_1-12_figure.png')
            st.image('Preprossed_Leads_13_figure.png')

        fig4 , ax4 = plt.subplots(4,3)
        fig4.set_size_inches(10, 10)
        x_counter=0
        y_counter=0
        for x,y in enumerate(Leads[:len(Leads)-1]):
            #converting to gray scale
            grayscale = color.rgb2gray(y)
            #smoothing image
            blurred_image = gaussian(grayscale, sigma=0.9)
            #thresholding to distinguish foreground and background
            #using otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            #creating binary image based on threshold
            binary_global = blurred_image < global_thresh
            #resize image
            binary_global = resize(binary_global, (300, 450))
            #finding contours
            contours = measure.find_contours(binary_global,0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
            if (x+1)%3==0:
                ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
                ax4[x_counter][y_counter].axis('image')
                ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
                x_counter+=1
                y_counter=0
            else:
                ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
                ax4[x_counter][y_counter].axis('image')
                ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
                y_counter+=1
    
            #scaling the data and testing
            lead_no=x
            scaler = MinMaxScaler()
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
            Normalized_Scaled=Normalized_Scaled.T
            #scaled_data to CSV
            if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
                Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
            else:
                Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)

        fig4.savefig('Contour_Leads_1-12_figure.png')
        my_expander3 = st.expander(label='CONOTUR LEADS')
        with my_expander3:
            st.image('Contour_Leads_1-12_figure.png')

        """#### **CONVERTING TO 1D SIGNAL**"""    
        #lets try combining all 12 leads
        test_final=pd.read_csv('Scaled_1DLead_1.csv')
        location= 'C://Users/My/Downloads/full_proj/WebWork/HeartDisease/'
        for files in natsorted(os.listdir(location)):
            if files.endswith(".csv"):
                if files!='Scaled_1DLead_1.csv':
                    df=pd.read_csv('C://Users/My/Downloads/full_proj/WebWork/HeartDisease/{}'.format(files))
                    test_final=pd.concat([test_final,df],axis=1,ignore_index=True)
    
        st.write(test_final)
        """#### **PASS TO ML MODEL FOR PREDICTION**"""
        loaded_model = joblib.load('ecg.pkl')
        result = loaded_model.fit_transform(test_final)

        if result[0] == 0:
            st.write("You ECG corresponds to Myocardial Infarction")
            predicted_result="You ECG corresponds to Myocardial Infarction"
    
        if result[0] == 1:
            st.write("You ECG corresponds to Abnormal Heartbeat")
            predicted_result="You ECG corresponds to Abnormal Heartbeat"
    
        if result[0] == 2:
            st.write("Your ECG is Normal")
            predicted_result="Your ECG is Normal"
    
        if result[0] == 3:
            st.write("You ECG corresponds to History of Myocardial Infarction")
            predicted_result="You ECG corresponds to History of Myocardial Infarction"
    
    return predicted_result

if __name__ == '__main__':
#Run the application
    app.run()  