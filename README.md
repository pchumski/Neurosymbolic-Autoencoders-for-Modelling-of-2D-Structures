readme_content = """# 🧠 Neurosymbolic Autoencoder for 2D Sequential and Grid Structures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Oficjalne repozytorium kodu dla pracy magisterskiej:** > *Autoenkodery neurosymboliczne dla modelowania sekwencyjnych struktur dwuwymiarowych* > **Politechnika Poznańska, Wydział Informatyki i Telekomunikacji (2026)**

Projekt stanowi implementację innowacyjnego systemu sztucznej inteligencji łączącego potęgę głębokich sieci neuronowych (percepcja) z interpretowalnością i rygorem matematycznym modeli symbolicznych (rozumowanie). System uczy się dekomponować złożone, przecinające się oraz silnie zaszumione struktury wizualne 2D na zbiór obiektowych prymitywów geometrycznych (elips), wykorzystując paradygmat **analysis-by-synthesis**.

---

## ✨ Kluczowe funkcjonalności (Features)

* 🔍 **Dwa alternatywne warianty dekodera:**
  * **Model Sekwencyjny (GRU):** Wykorzystuje sieci rekurencyjne ze wsparciem harmonogramu wygaszania mechanizmu wymuszania nauczyciela (*Teacher Forcing*) do dynamicznej generacji trajektorii obiektów i rozwiązywania problemu stopu.
  * **Przestrzenny Dekoder Siatkowy ($16\\times16$):** Autorska architektura rozwiązująca problem złożonych topologii (bifurkacje, rozgałęzienia, skrzyżowania typu X), eliminująca wpadanie w pętle nieskończone.
* 🎨 **Różniczkowalny Renderer 2D:** Wbudowany, analityczny silnik renderujący bazujący na funkcjach logistycznych (Sigmoid), zapewniający stabilny przepływ gradientu prosto z przestrzeni pikseli do parametrów elips.
* 🧬 **Generator danych "w locie":** Środowisko w pełni samowystarczalne. Generuje stochastyczne ścieżki i skomplikowane bifurkacje w pamięci RAM, aplikując ekstremalne degradacje wizualne (Gaussian Noise, Blur, Dropout), symulujące rzeczywiste obrazy medyczne (USG/Angiografia).
* 🚀 **Ortogonalna funkcja straty:** Rozwiązanie zapobiegające zjawisku *Zero-Collapse* poprzez całkowitą separację nauki topologii i geometrii (Współczynnik Dice'a) od fotometrii i barwy (MSE w przestrzeni RGB).
* 🎯 **100% Odtwarzalność (Reproducibility):** Sztywne zaszczepienie ziarna losowości (*Random Seed = 42*) gwarantuje identyczne zachowanie sieci i zbieżność krzywych uczenia przy każdym uruchomieniu.

---

## 📂 Architektura Repozytorium
```text
📦 Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures
 ┣ 📂 core
 ┃ ┣ 📜 renderer.py    # Matematyka różniczkowalnego renderera 2D
 ┃ ┗ 📜 losses.py      # Złożone, maskowane funkcje straty (Dice, MSE, BCE)
 ┣ 📂 engine
 ┃ ┗ 📜 trainer.py     # Pętle uczące (Standardowa oraz 3-fazowa dla GRU)
 ┣ 📂 models
 ┃ ┗ 📜 networks.py    # Enkoder CNN, Dekoder GRU oraz Dekoder Siatkowy
 ┣ 📂 utils
 ┃ ┣ 📜 dataset.py     # Generator danych (Geometric, Bifurcation, HardcoreUSG)
 ┃ ┗ 📜 visualization.py # Narzędzia analityczne (wykresy t-SNE, overlay)
 ┣ 📜 main.py          # Główny punkt wejścia CLI z obsługą argparse
 ┣ 📜 requirements.txt # Zależności środowiska uruchomieniowego
 ┗ 📜 README.md
