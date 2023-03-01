import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed, Dropout
from keras.layers import LSTM
import numpy as np
import keras
import numpy
import time

speaker_changed_seg = []
words = []

#Voice Activity Detection
import contextlib
import numpy as np
import wave
import librosa
import webrtcvad


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        #print(num_channels)
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        #print(sample_width)
        assert sample_width == 2
        sample_rate = wf.getframerate()
        #print(sample_rate)
        assert sample_rate in (8000, 16000, 32000, 48000)
        # print("dada shob assert pass kore")
        pcm_data = wf.readframes(wf.getnframes())
        #print(pcm_data)
        return pcm_data, sample_rate


class Frame(object):
  def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(vad, frames, sample_rate):
    is_speech = []
    for frame in frames:
        is_speech2=vad.is_speech(frame.bytes, sample_rate)
        is_speech.append(is_speech2)
    return is_speech


def vad(file):
    audio, sample_rate = read_wave(file)
    vad = webrtcvad.Vad(2)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(vad, frames, sample_rate)
    return segments

def speech(file):
  dummy = 0
  data = []
  segments = vad(file)
  audio, sr = librosa.load(file)
  for i in segments:
    if i == True:
      data.append(audio[dummy:dummy + 480])
      dummy = dummy + 480
    else:
      dummy = dummy + 480
  data = np.ravel(np.asarray(data))

  return data

def fxn(file):
  segments = vad(file)
  segments = np.asarray(segments)
  dummy = 0.01*np.where(segments[:-1] != segments[1:])[0] +.01 
  if len(dummy)%2==0:
    dummy = dummy
  else:
    dummy = np.delete(dummy, len(dummy)-1)

  voice = dummy.reshape(int(len(dummy)/2),2)
  
  return voice

# Commented out IPython magic to ensure Python compatibility.
#Segmentation (Each Segment will have only one Speaker)
# %tensorflow_version 2
import librosa
import matplotlib.pyplot as plt
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed, Dropout
from keras.layers import LSTM
import numpy as np
import keras

model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.build(input_shape=(None, 137, 35))
model.summary()
#Upload the pre-trained model file from Google Drive. Change the Path accordingly.
h5_model_file = 'C:/Users/HP/Downloads/di/model_bangla_2.h5'
model.load_weights(h5_model_file)


def multi_segmentation(file):
    frame_size = 2048
    frame_shift = 512
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    mfcc = mfccs[1:, ]
    norm_mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    norm_mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis=1, keepdims=True)) / np.std(mfcc_delta, axis=1, keepdims=True)
    norm_mfcc_delta2 = (mfcc_delta2 - np.mean(mfcc_delta2, axis=1, keepdims=True)) / np.std(mfcc_delta2, axis=1, keepdims=True)

    ac_feature = np.vstack((norm_mfcc, norm_mfcc_delta, norm_mfcc_delta2))
    #print(ac_feature.shape)

    sub_seq_len = int(3.2 * sr / frame_shift)
    sub_seq_step = int(0.8 * sr / frame_shift)

    def extract_feature():
        feature_len = ac_feature.shape[1]
        sub_train_x = []
        for i in range(0, feature_len-sub_seq_len, sub_seq_step):
            sub_seq_x = np.transpose(ac_feature[:, i: i+sub_seq_len])
            sub_train_x.append(sub_seq_x[np.newaxis, :, :])
        return np.vstack(sub_train_x), feature_len

    predict_x, feature_len = extract_feature()
    #print(predict_x.shape)

    predict_y = model.predict(predict_x)
    #print(predict_y.shape)

    score_acc = np.zeros((feature_len, 1))
    score_cnt = np.ones((feature_len, 1))

    for i in range(predict_y.shape[0]):
        for j in range(predict_y.shape[1]):
            index = i*sub_seq_step+j
            score_acc[index] += predict_y[i, j, 0]
            score_cnt[index] += 1

    score_norm = score_acc / score_cnt

    wStart = 0
    wEnd = 200
    wGrow = 200
    delta = 25

    store_cp = []
    index = 0
    while wEnd < feature_len:
        score_seg = score_norm[wStart:wEnd]
        max_v = np.max(score_seg)
        max_index = np.argmax(score_seg)
        index = index + 1
        if max_v > 0.5:
            temp = wStart + max_index
            store_cp.append(temp)
            wStart = wStart + max_index + 50
            wEnd = wStart + wGrow
        else:
            wEnd = wEnd + wGrow

    seg_point = np.array(store_cp)*frame_shift

    plt.figure('speech segmentation plot')
    plt.plot(np.arange(0, len(y)) / (float)(sr), y, "b-")

    for i in range(len(seg_point)):
        plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="c", linestyles="dashed")
        plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="r", linestyles="dashed")
    plt.xlabel("Time/s")
    plt.ylabel("Speech Amp")
    #plt.grid(True)
    plt.show()

    return np.asarray(seg_point) / float(sr)

#Re-segmentation (Based on Combining VAD and Segementation Output)
def group_intervals(a):
    a = a.tolist()
    ans = []

    curr = None
    for x in a:
        # no previous interval under consideration
        if curr == None:
          curr = x
        else:
            # check if we can merge the intervals
            if x[0]-curr[1] < 1:
                curr[1] = x[1]
            else:
            # if we cannot merge, push the current element to ans
                ans.append(curr)
                curr = x

        if curr is not None:
            ans.append(curr)

    d1 = np.asarray(ans)
    d2 = np.unique(d1)
    d3 = d2.reshape(int(len(d2)/2),2)
    return d3
    
def spliting(seg,arr):
  arr1 = arr.tolist()
  temp = arr.copy()
  
  for i in range(len(seg)-1):
    temp1 = float(seg[i])
    # print(temp1)
    #for j in range(len(arr)-1):
    for j in range(len(arr)):
      if ((temp1 > arr[j][0]) & (temp1 < arr[j][1])):
        arr1[j].insert(-1,(temp1))

  #for i in range(len(arr1-1)):
  for i in range(len(arr1)):
    size=len(arr1[i])
    if size>=3:
      arr1[i].pop(-2) #if arr1[i][-1]-arr1[i][-2]<0.2 else True
      
  return arr1
  
def final_reseg(arr):
  z=[]
  for i in arr:
    if len(i)==2:
      z.append(i)
    else:
      temp = len(i)
      for j in range(temp-1):
        if j!=temp-1:
          temp1 = [i[j],i[j+1]-0.01]
          z.append(temp1)
        elif j==temp-1:
          temp1 = [i[j],i[j+1]]
          z.append(temp1)
  
  return np.asarray(z)

# #Embedding Extraction


# import torch
# import librosa
# from pyannote.core import Segment

# def embeddings_(audio_path,resegmented,range):
#   model_emb = torch.hub.load('pyannote/pyannote-audio-master', 'emb')
 
#   embedding = model_emb({'audio': audio_path})
#   for window, emb in embedding:
#     assert isinstance(window, Segment)
#     assert isinstance(emb, np.ndarray)

#   y, sr = librosa.load(audio_path)
#   myDict={}
#   myDict['audio'] = audio_path
#   myDict['duration'] = len(y)/sr

#   data=[]
#   for i in resegmented:
#     excerpt = Segment(start=i[0], end=i[0]+range)
#     emb = model_emb.crop(myDict,excerpt)
#     data.append(emb.T)
#   data= np.asarray(data)
  
#   return data.reshape(len(data),512)

#modified embeddings

from pyannote.audio import Model,Inference
from pyannote.core import Segment
import librosa

def embeddings_(audio_path,resegmented,range):
    model = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
    inference = Inference(model,window="whole")
    y, sr = librosa.load(audio_path)
    myDict={}
    myDict['audio'] = audio_path
    myDict['duration'] = len(y)/sr
    data=[]
    for i in resegmented :
      excerpt = Segment(start=i[0], end=i[0]+2)
      embedding = inference.crop(myDict, excerpt)
      data.append(embedding.T)
    data= np.asarray(data)
    return data.reshape(len(data),512)

#Clustering (Mean-Shift)
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()
def clustering(emb):
  temp = scaler.fit_transform(emb)
  Y = TSNE(n_components=2).fit_transform(temp)
  cluster_ms = MeanShift(bandwidth = 3,max_iter=200,cluster_all=False).fit(Y)
  y_ms = cluster_ms.predict(Y)
  clus_centre = cluster_ms.cluster_centers_
  n_speakers = clus_centre.shape[0]
  # plt.figure
  # plt.scatter(Y[:,0], Y[:, 1], c=y_ms, s=50, cmap='viridis')
  # plt.show()

  return y_ms, n_speakers

# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# scaler = StandardScaler()
# def clustering(emb):
#   temp = scaler.fit_transform(emb)
#   Y = TSNE(n_components=2).fit_transform(temp)
#   kmeans = KMeans(n_clusters=7,init = 'k-means++',n_init=20, max_iter=500,algorithm='elkan')
#   kmeans.fit(Y)
#   y_kmeans = kmeans.predict(Y)
  
#   plt.figure
#   plt.scatter(Y[:,0], Y[:, 1], c=y_kmeans, s=50, cmap='viridis')
#   centers = kmeans.cluster_centers_
#   plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
#   plt.show()

#   return y_kmeans

#Generating Hypothesis
from pyannote.core import Annotation, Segment
def hypothesis_gen(hyp_df):
  hyp_records = hyp_df.to_records(index=False)
  hyp_rec = list(hyp_records)
  hypothesis = Annotation()
  i = 0
  # j = 0
  while i < (len(hyp_rec)-1):
    
    # if i < j:
    #   continue
    print(i)

    x= hyp_rec[i][1]
  
    while i < len(hyp_rec)-2 and hyp_rec[i][0] == hyp_rec[i+1][0]:
      y = hyp_rec[i][2]
      i = i+1
    y= hyp_rec[i][2]
        
    #print(i,x,y,"speaker: " + hyp_rec[i][0])
    speaker_changed_seg.append([hyp_rec[i][0],x,y])
    hypothesis[Segment(x, y)] = hyp_rec[i][0]
    i = i+1
    # if i > j:
    #   j=i

  

  return hypothesis

#Diarization 
from sklearn import preprocessing
import pandas as pd
@st.cache_data
def diarization(audiofile):
    voice = fxn(audiofile)
    segmented = multi_segmentation(audiofile)
    gp = group_intervals(voice)
    splt = spliting(segmented,gp)
    resegmented = final_reseg(splt)
    embeddings = embeddings_(audiofile,resegmented,2)
    speak_id , n_speakers = clustering(embeddings)
    label_list = []
    alpha = 'A'
    for i in range(0, n_speakers): 
        label_list.append(alpha) 
        alpha = chr(ord(alpha) + 1) 
    lb = preprocessing.LabelEncoder()
    label_hyp = lb.fit(label_list)
    speaker_id = lb.inverse_transform(speak_id)
    hyp_df = pd.DataFrame({'Speaker_id': speaker_id,'Offset (seconds)': resegmented[:, 0], 'end (seconds)': resegmented[:, 1]})
    result_hypo = hypothesis_gen(hyp_df)  
    return segmented, n_speakers, hyp_df, result_hypo

#Give the path of audio file for Speaker Diarization (It should be Mono type.)
#segmented, n_clusters, hyp_df, result_hypo = diarization('/content/drive/My Drive/SRU/commonsense.wav')
#print(n_speakers)
#print(hyp_df)
#result_hypo
# diarization('/content/drive/My Drive/SRU/BanglaSD_2.wav')


# Set app title
st.title('🎙️:blue[Bangla Speaker Diarization]')

# Add file upload widget
inputfile = st.file_uploader('Upload an audio file', type=['wav', 'mp3'])

# Add button to trigger speaker diarization
if st.button('Perform Speaker Diarization'):
    # Perform speaker diarization on the uploaded audio file
    #segmented, n_clusters, hyp_df, result_hypo = diarization('bangaleeMuslim_4.5.wav')
    segmented, n_clusters, hyp_df, result_hypo = diarization(inputfile.name)
    with st.spinner("Processing..."):
      time.sleep(3)
      st.markdown("**:green[Successfully Done!]**")
    n=str(n_clusters)
    # Display the predicted speaker labels
    st.header('Segmentation')
    st.dataframe(segmented, width=500)
    st.write('Number of Speakers: ', n)
    st.subheader(n)
    st.header('Speaker id with Start Time and End Time')
    st.dataframe(hyp_df, width=500)
    st.header('Predicted Speaker Labels')
    st.write(result_hypo)

    timeline = result_hypo.get_timeline()

    duration = [] 

    for segment in timeline:
      start_time = segment.start
      end_time = segment.end
      duration.append(end_time - start_time)

    fig1, ax1 = plt.subplots()
    ax1.pie(duration)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
    from streamlit_timeline import timeline
    #timeline=timeline(result_hypo)
    ##st.write(timeline)
    #st.line_chart(fig1)
    # st.bar_chart(fig1)
    #st.area_chart(fig1)
    #st.line_chart(result_hypo)
    #st.cache_data.clear()

