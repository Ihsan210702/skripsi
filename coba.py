import streamlit as st

# Fungsi untuk membuat tautan
def tampilkan_tautan(judul, url):
    return f'<a href="{url}" targt="_target">{judul}</a>'

# URL dan judul artikel
url_artikel_1 = "/main.py"
judul_artikel_1 = "main.py"
url_artikel_2 = "URL_ARTIKEL_2"
judul_artikel_2 = "Judul Artikel 2"

# Tampilkan tautan untuk artikel 1
st.markdown(tampilkan_tautan(judul_artikel_1, url_artikel_1), unsafe_allow_html=True)

# Tampilkan tautan untuk artikel 2
st.markdown(tampilkan_tautan(judul_artikel_2, url_artikel_2), unsafe_allow_html=True)
