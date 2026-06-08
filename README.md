# 🧠 Neurosymbolic Autoencoder for 2D Sequential Structures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Oficjalne repozytorium kodu dla pracy magisterskiej:** > *Autoenkodery neurosymboliczne dla modelowania sekwencyjnych struktur dwuwymiarowych.*

Projekt to innowacyjny system sztucznej inteligencji, który łączy w sobie potęgę głębokich sieci neuronowych (percepcja) z interpretowalnością modeli symbolicznych (rozumowanie). System uczy się dekomponować złożone, przecinające się i zaszumione struktury wizualne 2D na zbiór obiektowych prymitywów geometrycznych, wykorzystując paradygmat **analysis-by-synthesis**.

---

## ✨ Kluczowe funkcjonalności (Features)

* 🔍 **Dwie Architektury Dekodera:** * **Model Sekwencyjny (GRU):** Wykorzystuje sieci rekurencyjne ze wsparciem wymuszania nauczyciela (*Teacher Forcing*) do rozwiązywania problemu stopu (zmiennej liczebności obiektów).
    * **Przestrzenny Dekoder Siatkowy ($16\times16$):** Autorska odpowiedź na problem skomplikowanych topologii (bifurkacje, skrzyżowania typu X), całkowicie omijająca wpadanie w pętle.
* 🎨 **Różniczkowalny Renderer 2D:** Wbudowany analityczny silnik renderujący bazujący na funkcjach logistycznych (Sigmoid), umożliwiający wsteczną propagację gradientu prosto z przestrzeni pikseli do parametrów elips.
* 🧬 **Generator Danych "w locie":** Środowisko nie wymaga pobierania gigabajtów danych. Dataset tworzy wielokrotnie przecinające się struktury w locie, aplikując ekstremalne degradacje wizualne (Gaussian Noise, Blur, Dropout).
* 🚀 **Ortogonalna Funkcja Straty:** Innowacyjne rozwiązanie zapobiegające zjawisku *Zero-Collapse* poprzez separację nauki topologii (Współczynnik Dice'a) od fotometrii (MSE przestrzeni RGB).

---

## 📂 Architektura Repozytorium

```text
📦 Neurosymbolic-Autoencoder
 ┣ 📂 core
 ┃ ┣ 📜 renderer.py    # Matematyka różniczkowalnego renderera 2D
 ┃ ┗ 📜 losses.py      # Złożone, maskowane funkcje straty (Dice, MSE, BCE)
 ┣ 📂 engine
 ┃ ┗ 📜 trainer.py     # Pętle uczące, optymalizator Adam, harmonogramy
 ┣ 📂 models
 ┃ ┗ 📜 networks.py    # Enkoder CNN, Dekoder GRU oraz Dekoder Siatkowy
 ┣ 📂 utils
 ┃ ┣ 📜 dataset.py     # Stochastyczny generator danych geometrycznych i szumu
 ┃ ┗ 📜 visualizer.py  # Narzędzia analityczne (wykresy t-SNE, overlay)
 ┣ 📜 main.py          # Główny punkt wejścia do trenowania modelu
 ┣ 📜 requirements.txt # Zależności środowiska
 ┗ 📜 README.md
