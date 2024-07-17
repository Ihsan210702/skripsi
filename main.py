import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import pickle5 as pickle 
# from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

df1 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/steaming.csv')
with open('slangdict.txt') as f:
        data = f.read()
slang_dict = json.loads(data)

#membuat fungsi untuk input
def case_folding(input):
        return input.lower()
def cleaning(input):
        clean = re.sub("[^a-zA-Zï ]+"," ", input)
        return clean
def tokenize(input):
        return nltk.word_tokenize(input)
def replace_slang_words(input):
        replaced_words = [slang_dict[words] if words.lower() in slang_dict else words for words in words]
        return replaced_words
def stopword(input):
        stop_factory = StopWordRemoverFactory()
        stopwords_nltk = set(stopwords.words('indonesian'))
        filtered_words = [word for word in words if word not in stopwords.words('indonesian')]
        return filtered_words
def stemming(input):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = stemmer.stem(input)
        return stem


# # Memuat model dari file pickle
# with open("adaboost_model.pkl", "rb") as f:
#     adaboost_model_loaded = pickle.load(f)

df = df1["Steaming"].fillna(' ')
tfidfvectorizer = TfidfVectorizer()
tfidf_wm = tfidfvectorizer.fit_transform(df)
tfidf_tokens = tfidfvectorizer.get_feature_names_out()
labels = df1['Label']
# Membuat peta/kamus label
label_mapping = {"positif": 0, "negatif": 1, "campuran": 2}
# Mengubah label menjadi nilai numerik secara manual
y_numeric = [label_mapping[label] for label in labels]
#
# undersample=RandomOverSampler(random_state=1)
# X_train_ros, y_train_ros = undersample.fit_resample(tfidf_wm, y_numeric)
#Train test split
training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
training_label, test_label = train_test_split(y_numeric, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

#model
model = MultinomialNB()
adaboost_model = AdaBoostClassifier(base_estimator=model, n_estimators=300)
clf = adaboost_model.fit(training, training_label)


#implementasi dari sistem
tab1, tab2, tab3, tab4,tab5 = st.tabs(["🏠 Home", "💹 Implementasi","🗃 Dataset", "Naive Bayes", "Adaptive Boosting"])

with tab1:
        st.title("Analisis Sentimen Terhadap Pariwisata Madura Menggunakan Naive Bayes dan Adaptive Boosting")
        st.write("""<h3 style = "text-align: center;">
        <img src="https://images.tokopedia.net/blog-tokopedia-com/uploads/2019/08/wisata-madura-1-Native-Indonesia.jpg" border="0" width="700" height="370">
        </h3>""",unsafe_allow_html=True)
        st.write(""" """)
        st.write("Analisis sentimen adalah teknik yang membantu mengidentifikasi dan mengekstrak informasi subjektif, seperti opini tentang suatu artikel.Tujuan dari analisis sentimen adalah untuk memisahkan dokumen yang berupa kata atau kalimat menjadi sentimen positif, negatif ataupun netral.")
with tab2:
        st.title("Implementasi Prediksi Sentimen")
        # st.subheader("Masukkan ulasan yang akan diprediksi")
        text = st.text_input("Masukkan ulasan yang akan diprediksi")
        periksa = st.button("Periksa")
        if periksa:
                if text == "":
                        st.warning("Silahkan Masukkan Inputan Teks")
                else:
                        hasil = case_folding(text)
                        clean = cleaning(hasil)
                        words = tokenize(clean)
                        slang = replace_slang_words(words)
                        tokens = stopword(slang)
                        text = ' '.join(tokens)
                        stem = stemming(text)
                        v_data = tfidfvectorizer.transform([stem]).toarray()
                        # tfidf_tokens = tfidfvectorizer.get_feature_names_out()
                        # df_tfidfvect = pd.DataFrame(data = v_data,columns = tfidf_tokens)
                        # df_tfidfvect
                        y_preds = clf.predict(v_data)
                        st.write("Hasil Preprocessing : ", stem)
                        st.subheader('Prediksi')
                        if y_preds == 0:
                                st.success('Positif')
                        elif y_preds == 1:
                                st.error('Negatif')
                        elif y_preds == 2:
                                st.warning('Campuran')
                        else:
                                st.error('Salah')
with tab3:
        st.header("Dataset")
        #Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/Data_Pariwisata.csv')
        st.dataframe(df, use_container_width=True) 

        st.header("Data Hasil Preprocessing")
        #Dataset
        df1 = df1.drop(columns='Unnamed: 0')
        st.dataframe(df1, use_container_width=True) 
        label_counts = df1['Label'].value_counts(dropna=False)
        st.subheader("Distribusi Label")
        # Plot the counts as a bar chart
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed
        label_counts.plot(kind='bar', ax=ax, color='skyblue')
        # ax.set_title('Distribusi Label')
        ax.set_xlabel('Label')
        ax.set_ylabel('Counts')
        # st.write("Distribusi Label:")
        st.write(label_counts)
        # Display the plot
        st.pyplot(fig)
        st.subheader("Distribusi Label setelah Random Under Sampling")
        undersample=RandomUnderSampler(random_state=1)
        X_train_ros, y_train_ros = undersample.fit_resample(tfidf_wm, y_numeric)
        label_mapping = {0: "positif", 1: "negatif", 2: "campuran"}
        y_train_ros_categories = [label_mapping[label] for label in y_train_ros]
         # Get the counts of each class after undersampling
        counter = Counter(y_train_ros_categories)
        st.write(pd.DataFrame.from_dict(counter, orient='index', columns=['Counts']))
        # Plot the counts as a bar chart
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed
        ax.bar(counter.keys(), counter.values(), color='yellow')  # Change the color as needed
        # ax.set_title('Distribusi Label setelah Random Under Sampling')
        ax.set_xlabel('Label')
        ax.set_ylabel('Counts')
        
        # Display the plot
        st.pyplot(fig)
       

        st.subheader("Distribusi Label setelah Random Over Sampling")
        undersample=RandomOverSampler(random_state=1)
        X_train_ros, y_train_ros1 = undersample.fit_resample(tfidf_wm, y_numeric)
        label_mapping1 = {0: "positif", 1: "negatif", 2: "campuran"}
        y_train_ros_categories1 = [label_mapping1[label] for label in y_train_ros1]
        conter = Counter(y_train_ros_categories1)
        st.write(pd.DataFrame.from_dict(conter, orient='index', columns=['Counts']))
        # Plot the counts as a bar chart
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed
        ax.bar(conter.keys(), conter.values(), color='red')  # Change the color as needed
        # ax.set_title('Distribusi Label setelah Random Under Sampling')
        ax.set_xlabel('Label')
        ax.set_ylabel('Counts')
        
        # Display the plot
        st.pyplot(fig)
        
# with tab4:
#         st.header("Hasil Akurasi")

with tab4:
        st.header("Perhitungan Naive Bayes")
        st.subheader("1. Menghitung nilai prior masing-masing kelas")
        with open('model_naive_bayes.pkl', 'rb') as f:
            class_tf_idf, class_word_count, prior_probabilities, likelihood = pickle.load(f)
        # Menampilkan probabilitas prior untuk setiap kelas
        st.write("Probabilitas Prior untuk Setiap Kelas:")
        for label, prob in prior_probabilities.items():
                st.write(f'Kelas {label}: {prob}')

        st.subheader("2. Menghitung nilai probabilitas pada setiap kata pada masing-masing kelas")
        fitur_prob = pd.read_csv("fitur_prob.csv")
        st.dataframe(fitur_prob, use_container_width=True) 

        st.subheader("3. Menghitung nilai likelyhood pada setiap kata pada masing-masing kelas")
        likelihood = pd.read_csv("likelihood.csv")
        likelihood
        # st.dataframe(likelihood, use_container_width=True) 
with tab5:
        st.header("Perhitungan Adaboost")
        st.subheader("1. Menghitung nilai error pada setiap iterasi")
        likelihood = pd.read_csv("estimator_error.csv")
        st.dataframe(likelihood, use_container_width=True) 
        import numpy as np
        # Menghitung estimated error dan jumlah data yang salah diprediksi untuk setiap estimator dalam AdaBoost
        # estimator_errors = []
        # estimator_misclassified_counts = []
        # for i, estimator in enumerate(clf.estimators_):
        # y_pred = estimator.predict(training)
        # incorrect = (y_pred != training_label)
        # estimator_error = np.mean(incorrect)
        # misclassified_count = np.sum(incorrect)
        # estimator_errors.append(estimator_error)
        # estimator_misclassified_counts.append(misclassified_count)

        # # Menampilkan estimated error dan jumlah data yang salah diprediksi dari setiap estimator
        # for i, (error, count) in enumerate(zip(estimator_errors, estimator_misclassified_counts)):
        #         print(f'Estimator {i+1}: Estimated Error = {error}, Misclassified Count = {count}')
       
