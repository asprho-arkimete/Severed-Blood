# Severed Blood

Videogame horror/dark style ispirato a **Hostel**.

## Scopo del gioco

Riempire l'ampolla di sangue prelevando il sangue dalle vittime.

- Carica 4 personaggi (consigliati femminili per via del prompt generation delle immagini)
- Raccogli il loro sangue generando immagini in stile slasher
- Oggetti a disposizione: sega, ferro, martello, cacciavite
- Sedia per fare pose fotografiche

## Strumenti utilizzati

- Flux 2 Klein 9B
- 8 livelli generati casualmente

## Installazione

1. Installa **Python 3.10**
2. Installa **Anaconda**
3. Installa **Cursor**

Clona il repository:

```bash
git clone https://github.com/asprho-arkimete/Severed-Blood.git
cd Severed-Blood
```

Installa CUDA (PyTorch):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Scarica dalla sezione **Releases**:
- `NextLevel.mp4`
- le parti del file LoRA (`.rar`)

e posizionale nella directory principale (`main`).

Decomprimi tutti i file scaricati.

Avvia il gioco:

```bash
python severed.py
```

## Come giocare

All'avvio, il gioco creerà automaticamente tutte le location necessarie.

Porta i 4 personaggi nelle canvas e premi **Avvia gioco** per iniziare la partita.


