
import tkinter as tk
from tkinter import HORIZONTAL, ttk
from httpx import delete
import torch
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageTk
import os
import shutil
from tqdm import tqdm
from tkinterdnd2 import TkinterDnD, DND_FILES

# immortalita
immortalita = False  # ⚠️ TRUCCO DI TEST — ricordati di rimettere False prima del rilascio!

if os.path.exists("XR$%&(JHHGFRRTGGDFD)()KJJHB.bat"):
    immortalita=True

if immortalita:
    print("⚠️ ATTENZIONE: modalità immortalità attiva, i personaggi non possono morire!")

canvas_w, canvas_h = 1408, 800  # 1408 = 88*16, 800 = 50*16
DEFAULT_STEPS = 8
out_dir = "location"
path_image1 = None
path_image2 = None
path_image3 = None
path_image4 = None
path_lora = None

CHAR_DIR = "./character"

dc = 128       # dimensione canvas personaggio
border = 5     # spessore cornice

os.makedirs(CHAR_DIR, exist_ok=True)

import os
import datetime
import torch
import traceback

def flux2(prompt, steps=DEFAULT_STEPS, path_image1=None, path_image2=None, path_image3=None,
          path_image4=None, path_lora=None, wc=None, hc=None, out_dir=out_dir, name=None):
    global canvas_w, canvas_h
    print("flux 2 generazione immagine")

    # conserva il path/nome ORIGINALE prima di eventuali sovrascritture,
    # cosi il nome del file di output resta coerente con la stanza di partenza
    path_image1_originale = path_image1

    if path_image1 is not None and os.path.exists("./disegno.png"):
        data_modifica = datetime.date.fromtimestamp(os.path.getmtime("./disegno.png"))
        if data_modifica == datetime.date.today():
            print("APPLICO LO SKIZZO DEL IMAGE DISEGNO SULLO SFONDO STANZA")
            with Image.open(path_image1).convert("RGBA") as img_stanza, \
                 Image.open("./disegno.png").convert("RGBA") as img_disegno:

                # incolla il disegno sopra lo sfondo, usando l'alpha come maschera
                # cosi le zone trasparenti non sovrascrivono lo sfondo
                img_stanza.paste(img_disegno, (0, 0), img_disegno)
                img_stanza.save("stanza_skizzo.png")

            path_image1 = "./stanza_skizzo.png"

    # se non specificata, usa la risoluzione canvas (caso sfondi)
    if wc is None:
        wc = canvas_w
    if hc is None:
        hc = canvas_h

    device = "cuda"
    dtype = torch.bfloat16
    repo = "black-forest-labs/FLUX.2-klein-9B"

    try:
        print("📦 Caricamento modello...")
        pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=dtype)

        if path_lora and os.path.exists(path_lora):
            print("📎 Caricamento LoRA...")
            pipe.load_lora_weights(path_lora, adapter_name='lora')
            pipe.set_adapters(["lora"], adapter_weights=[0.8])
            print(f"path lora: {path_lora}")

        print("⚡ Quantizzazione...")
        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)

        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)

        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            quantize(pipe.text_encoder_2, weights=qfloat8)
            freeze(pipe.text_encoder_2)

        pipe.enable_model_cpu_offload()

        print(f"🌍 Traduzione: {prompt}")
        prompt_eng = GoogleTranslator(source='it', target='en').translate(prompt)
        print(f"✅ Prompt EN: {prompt_eng}")

        def f_resize(path_image, rw, rh):
            with Image.open(path_image) as img:
                img = img.convert("RGB")
                w, h = img.size
                if w >= h:
                    rh = (rw * h) // w
                else:
                    rw = (rh * w) // h
                return img.resize((rw, rh), Image.BICUBIC)

        images = []
        for p in (path_image1, path_image2, path_image3, path_image4):
            if p and os.path.exists(p):
                images.append(f_resize(p, 256, 256))

        print(f"img1: {path_image1}")
        print(f"img2: {path_image2}")
        print(f"img3: {path_image3}")
        print(f"img4: {path_image4}")

        gen_kwargs = dict(
            prompt=prompt_eng,
            height=hc,
            width=wc,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(0)
        )
        if images:
            gen_kwargs["image"] = images[0] if len(images) == 1 else images

        print("🎨 Generazione immagine...")
        image = pipe(**gen_kwargs).images[0]
        print("✅ Immagine generata!")

        # ✅ SALVATAGGIO CON GESTIONE ERRORI
        try:
            print(f"💾 Creazione directory: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            if name is None:
                if path_image1_originale and os.path.exists(path_image1_originale):
                    base_name = os.path.splitext(os.path.basename(path_image1_originale))[0]
                else:
                    base_name = "output"
                name = f"{base_name}_flux"

            file_path = os.path.join(out_dir, f"{name}.png")
            print(f"📁 Percorso salvataggio: {file_path}")

            image.save(file_path)
            print(f"✅ Immagine salvata: {file_path}")

        except Exception as e:
            print(f"❌ ERRORE SALVATAGGIO: {e}")
            print(f"❌ Type: {type(e)}")
            traceback.print_exc()
            return None

        # ✅ PULIZIA MEMORIA
        try:
            print("🧹 Pulizia memoria GPU...")
            del pipe
            torch.cuda.empty_cache()
            print("✅ Memoria pulita")
        except Exception as e:
            print(f"⚠️ Errore pulizia: {e}")

        return image

    except Exception as e:
        print(f"❌ ERRORE GENERAZIONE: {e}")
        print(f"❌ Type: {type(e)}")
        traceback.print_exc()
        return None

save_livel = "./livel.txt"
Livel = 1
if os.path.exists(save_livel):
    with open(save_livel) as f:
        line = f.readline()
        try:
            Livel = int(line.strip())
        except ValueError:
            Livel = 1
    
# stine horror classico

PROMPT1=''
PROMPT2=''
PROMPT3=''
PROMPT4=''
PROMPT5=''
PROMPT6=''


def prompts():
    global Livel,PROMPT1,PROMPT2,PROMPT3,PROMPT4,PROMPT5,PROMPT6
    print(f"prompt livel: {Livel}")

    if Livel == 1:
        #attrezzi    
        PROMPT1 = """
        Un'atmosfera cupa e angosciante da officina/sala chirurgica improvvisata in stile
        horror, illuminata debolmente da una singola lampadina a incandescenza nuda che pende
        da un cavo logoro al centro del soffitto. La stanza ha pareti e soffitto in cemento
        scrostato e sporco, con macchie di umidità e mattoni visibili sulle pareti laterali.

        Sulla parete di fondo, un grande pannello perforato in legno mostra un'ampia varietà
        di utensili e lame arrugginiti e usurati: un seghetto, bisturi, pinze, forbici, un
        vecchio trapano manuale, diversi coltelli da cucina e da caccia di varie dimensioni,
        pinzette e attrezzi metallici assortiti, tutti appesi ordinatamente ma invecchiati e
        macchiati.

        In primo piano, un robusto banco di lavoro in legno graffiato domina la scena, coperto
        da profonde scalfitture, macchie di sangue secco e chiodi arrugginiti sparsi sulla
        superficie. Diverse fessure rettangolari sono tagliate nel piano del tavolo. Sul
        tavolo: un grosso tronchese/tagliaossa arrugginito al centro-sinistra, uno straccio
        sporco macchiato di sangue a destra, e un singolo bisturi nell'angolo in basso a
        destra. Due stracci più piccoli piegati si trovano più indietro vicino alla parete
        del pannello, insieme a una barra metallica.

        Atmosfera: oscura, inquietante, illuminazione a basso contrasto con forte contrasto
        tra la lampadina luminosa e le ombre profonde, atmosfera polverosa e decadente,
        estetica da gioco horror, fotorealistico, cinematografico, texture molto detagliate
        su legno, metallo e cemento, leggera desaturazione, singola fonte di luce drammatica
        che proietta ombre dure verso il basso.

        Camera: a livello degli occhi, centrata, composizione simmetrica, prospettiva grandangolare
        guardando direttamente verso il banco di lavoro e la parete degli attrezzi.

        Mantieni massima coerenza del ambientazione e degli oggetti di scena e non cambiare inquadratura
        Mantieni stile dark horror poca illuminazione"""

        #bagno
        PROMPT2 = """
        Bagno abbandonato in stile horror, ambientazione notturna, render 3D cinematografico. Composizione centrata con una toilette sporca in primo piano, coperchio alzato, water riempito di liquido rossastro scuro simile a sangue, increspato con cerchi concentrici, con una sostanza scura e densa simile a catrame che fuoriesce dal bordo e cola sul pavimento. Il sedile e il serbatoio del water sono macchiati con schizzi color ruggine.
        Dietro la toilette, una grande finestra a vetri multipli con diversi pannelli di vetro rotti/frantumati rivela un cielo notturno cupo con luna piena e sagome di rami d'albero spogli e senza foglie. Una luce lunare blu-grigia freddo filtra dalla finestra, fungendo da fonte di luce principale.
        A sinistra, uno specchio ovale incrinato con un riflesso distorto in stile horror, appeso sopra un lavabo a colonna vintage con rubinetteria d'epoca, rigato di sangue che scende lungo il muro e il lavandino. A destra, una vasca da bagno vintage con piedi a zampa di leone, parzialmente in ombra, con liquido denso e scuro che si accumula e cola lungo il fianco fino al pavimento piastrellato.
        Pavimento coperto da piastrelle rotte e crepate, grandi pozze di sangue secco e pozze di liquido nero viscoso che si spandono sulle piastrelle, piccoli frammenti di ossa sparsi visibili. Pareti coperte da carta da parati vintage a motivi damascati sbiadita, pesantemente macchiata di schizzi e colature di sangue, con texture di danni causati dall'acqua.
        Illuminazione: cupa, suggestiva, tonalità desaturate e fredde (palette blu-grigia) in contrasto con toni rosso/ruggine intensi del sangue e ombre nere profonde. Forte luce direzionale proveniente dalla finestra che crea un alto contrasto tra le aree illuminate e quelle in ombra. Atmosfera horror inquietante, texture fotorealistiche, sporco e degrado ultra-dettagliati, estetica cinematografica da videogioco horror, inquadratura architettonica grandangolare.
        Mantieni massima coerenza del ambientazione e poca illuminazione
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto in stile horror/decadente, bagnata in una luce rossa intensa e inquietante, render 3D fotorealistico. Una singola lampadina nuda accesa di colore rosso acceso pende da un cavo elettrico attorcigliato dal soffitto, fungendo da unica fonte di luce della stanza e tingendo l'intero ambiente di tonalità rosse e bordeaux.
        In primo piano, un letto disfatto con lenzuola e cuscini rossi spiegazzati, sulla cui superficie sono sparsi bottiglie di vetro scuro vuote (di alcolici), riviste e giornali aperti. Il pavimento di legno è coperto da altre bottiglie rovesciate, pagine di giornale sparse e riviste abbandonate, suggerendo abbandono, dipendenza o disagio mentale.
        A sinistra, una finestra con tende pesanti scostate, vetri scuri che non lasciano intravedere nulla all'esterno, e un comò in legno con un piccolo posacenere acceso (con una brace o sigaretta) sopra. A destra, un comodino in legno con un cassetto cassettiera aperto e vuoto, sormontato da un posacenere pieno di mozziconi di sigaretta accesi, e dietro di esso uno scaffale/libreria in legno con un cassetto leggermente aperto.
        Le pareti sono rovinate, scrostate, con crepe profonde e vistose che si diramano come ragnatele su tutta la superficie, intonaco scrostato e macchie di umidità, contribuendo a un'atmosfera di degrado e abbandono.
        Illuminazione: dominante rossa intensa e satura proveniente dalla lampadina centrale, che crea forti contrasti tra zone illuminate di rosso vivido e ombre profonde nere/bordeaux. Atmosfera claustrofobica, inquietante, opprimente, tipica di un ambiente horror psicologico o di un luogo legato a violenza/disagio. Texture fotorealistiche dei tessuti, del legno invecchiato e delle pareti scrostate, dettagli ultra-definiti su crepe, macchie e oggetti sparsi. Stile cinematografico da horror game, inquadratura prospettica dal basso verso il letto.
        Mantieni massima coerenza del ambientzione e della poca illuminazione
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata e devastata, ambientazione notturna molto buia, render 3D fotorealistico in stile horror/abbandono. Composizione simmetrica centrata su una finestra a doppio battente sul fondo, con vetri sporchi e macchiati da cui filtra una debole luce blu-grigia fredda, unica fonte di illuminazione della stanza, che crea un effetto di raggi di luce soffusa (volumetric light) che penetra nella stanza buia.
        Al centro, sotto la finestra, un lavello da cucina vintage in ceramica bianca pieno di piatti sporchi accatastati, con un canovaccio giallo appeso al bordo. A destra del lavello, una vecchia cucina/fornello a gas bianco con sportello del forno, macchiato e arrugginito, con colature scure che scendono lungo la superficie. Sopra il lavello e ai lati della finestra, pensili e mobili da cucina scrostati e malridotti, con sportelli aperti e cassetti fuori sede, alcuni vuoti, altri con bottiglie di vetro scuro disposte sopra.
        Il pavimento in primo piano è interamente ricoperto di disordine: numerose bottiglie di vetro scuro (alcolici) vuote sparse e rovesciate, lattine di birra schiacciate, mozziconi di sigaretta sparsi ovunque, bicchieri rotti e frammenti di vetro, piatti rotti, e numerosi giornali e riviste sparpagliati sul pavimento. Al centro in primo piano, una grande pozza d'acqua/liquido che riflette debolmente la luce della finestra, sopra pagine di giornale bagnate e macchiate.
        Pareti scrostate e ammuffite ai lati, immerse quasi completamente nell'oscurità, con i mobili della cucina a sinistra e a destra che si dissolvono nel buio profondo ai margini dell'inquadratura.
        Illuminazione: scena estremamente scura e cupa, dominata da ombre nere profonde, con un'unica fonte di luce blu-grigia fredda e soffusa proveniente dalla finestra centrale sul fondo, che illumina debolmente il lavello e crea un forte contrasto chiaroscurale con il resto della stanza immerso nel buio. Atmosfera inquietante, claustrofobica, decadente, tipica di un survival horror, polvere e particelle in suspensione visibili nel fascio di luce. Texture fotorealistiche ultra-dettagliate su superfici sporche, oggetti abbandonati, carta consumata e macchie di sporco. Stile cinematografico da videogioco horror, inquadratura frontale e simmetrica con prospettiva centrale verso la finestra.
        Mantieni massima coerenza del ambientzione e della poca illuminazione
        """

        #teatro
        PROMPT5 = """
        Interno di un antico teatro all'italiana, vista da un palco/balconata laterale verso il sipario centrale, render 3D fotorealistico, atmosfera elegante ma cupa e misteriosa. In primo piano, pavimento in parquet di legno a intarsio geometrico (motivo a scacchiera con pannelli quadrati alternati), lucido e riflettente, illuminato da un fascio di luce soffusa che crea un punto luminoso centrale sul pavimento.
        Ai lati, due grandi colonne/pilastri in legno massiccio scuro, con basamenti decorativi scolpiti, sormontate da mensole in metallo nero con diversi gancetti/appendini in ferro battuto (probabilmente un vecchio guardaroba o appendiabiti teatrale). Le colonne incorniciano la scena creando una composizione simmetrica.
        Sullo sfondo centrale, un grande sipario teatrale di velluto rosso bordeaux, con drappeggi a festoni nella parte superiore bordati di frange dorate, racchiuso da una cornice scenica dorata ornata con decorazioni scolpite (motivi floreali e voluta centrale dorata in alto). Davanti al sipario, un piccolo palco/pedana rialzata.
        Tra il primo piano e il fondale, una balaustra in legno tornito con colonnine decorative, che separa la balconata su cui ci si trova dalla platea sottostante, dove si intravedono file di poltrone rosse in velluto. Ai lati, parzialmente visibili nell'ombra, palchi laterali (gallerie) con balaustre dorate e velluto rosso, tipici di un teatro storico ottocentesco.
        Illuminazione: scena generale in penombra con toni caldi marroni e dorati, illuminazione drammatica e soffusa che crea un forte contrasto tra le zone in ombra (colonne in primo piano, palchi laterali) e il punto luminoso centrale sul sipario e sul pavimento. Atmosfera solenne, vintage, leggermente inquietante e misteriosa, tipica di un teatro abbandonato o silenzioso fuori orario di spettacolo. Texture fotorealistiche ultra-dettagliate sul legno (venature, intarsi del parquet), sul velluto del sipario e sui metalli anticati dei gancetti. Stile cinematografico, prospettiva centrale simmetrica con punto di fuga verso il sipario.
        Mantieni massima coerenza del ambientzione e della poca illuminazione
        """

        #portone
        PROMPT6 = """
        Scena horror/dark fantasy con un grande portone metallico a due ante in primo piano, su sfondo completamente buio, render 3D fotorealistico. Il portone è realizzato in metallo arrugginito e ossidato, con grandi pannelli rettangolari rivettati (bulloni/rivetti metallici disposti lungo i bordi e le giunture), texture di ruggine color bruno-rossastro che si dirama in macchie irregolari sulla superficie, con colature verticali simili a sangue secco/ruggine che scendono lungo le ante.
        Al centro del portone, una grande sbarra/trave metallica orizzontale funge da sistema di blocco, attraversando entrambe le ante a circa un terzo dall'alto, fissata con anelli metallici a entrambe le estremità. Da questi due anelli pendono delle catene metalliche che sostengono una grande bilancia a doppio piatto (simbolo della giustizia), con due bacinelle/ciotole metalliche rotonde e profonde che pendono simmetricamente ai lati, leggermente sbilanciate, creando un forte elemento simbolico e inquietante al centro della composizione.
        Sullo sfondo, in alto, un piccolo cono di luce bianca fredda e polverosa illumina debolmente la parte superiore della scena, creando un effetto di luce volumetrica con particelle di polvere sospese in aria, mentre il resto della scena rimane immerso in un'oscurità quasi totale. Ai lati e in basso, ragnatele fitte e dense si estendono dagli angoli verso il centro, suggerendo abbandono prolungato.
        Il pavimento in primo piano è in pietra/cemento grezzo, parzialmente visibile sotto le bilance, anch'esso scarsamente illuminato. Le pareti laterali, a malapena visibili, mostrano crepe profonde nell'intonaco scuro.
        Illuminazione: scena drammatica con altissimo contrasto, dominata da un buio quasi assoluto (nero profondo) interrotto solo da una sottile luce dall'alto che crea un'atmosfera cupa, claustrofobica e minacciosa. Le tonalità sono fredde e desaturate nella parte alta (luce bianco-bluastra) in contrasto con i toni calde ruggine/marrone del portone metallico. Atmosfera horror gotico, simbolica e inquietante, tipica di un ingresso a un luogo di giudizio o prigione. Texture fotorealistiche ultra-dettagliate su metallo arrugginito, catene, ragnatele e pietra. Stile cinematografico, inquadratura frontale e perfettamente simmetrica.
        Mantieni massima coerenza del ambientzione e della poca illuminazione
        """

    elif Livel == 2:
        # stile neve invernale — palette fredda bianco/blu ghiaccio, il sangue diventa quasi nero contro la neve
        #attrezzi
        PROMPT1 = """
        Officina/sala chirurgica improvvisata sepolta in un rifugio invernale abbandonato, stile horror,
        illuminata da una singola lampadina nuda che pende dal soffitto, la sua luce fredda e biancastra
        riflessa da un sottile strato di brina che ricopre ogni superficie. Pareti in cemento scrostato
        coperte da patina di ghiaccio e stalattiti di ghiaccio agli angoli, spifferi gelidi che sollevano
        piccole nuvole di neve polverizzata visibili nel fascio di luce.

        Sulla parete di fondo, il pannello perforato in legno mostra gli stessi utensili arrugginiti —
        seghetto, bisturi, pinze, forbici, trapano manuale, coltelli — ma ora incrostati di brina e
        ghiaccio sottile, con piccoli ghiaccioli che pendono dai bordi delle lame.

        Il banco di lavoro in legno graffiato è coperto da un velo di neve leggera, con il sangue secco
        che appare quasi nero-violaceo contro il bianco, e chiodi arrugginiti che spuntano dal ghiaccio.
        Stesso tronchese/tagliaossa al centro-sinistra, ora semi-ghiacciato, straccio rigido e congelato
        a destra, bisturi in basso a destra circondato da cristalli di ghiaccio.

        Atmosfera: gelida, silenziosa, desaturata su toni blu-bianco-grigio, respiro visibile se ci fosse
        un personaggio, luce fredda e tagliente, stessa composizione simmetrica e stesso banco frontale.
        Fotorealistico, cinematografico, texture di ghiaccio e brina ultra-dettagliate su legno e metallo.

        Camera: identica inquadratura a livello degli occhi, centrata, grandangolare verso banco e parete attrezzi.
        Mantieni massima coerenza dell'ambientazione e degli oggetti di scena, cambia solo palette fredda e presenza di ghiaccio/neve.
        """

        #bagno
        PROMPT2 = """
        Bagno abbandonato ricoperto di neve e ghiaccio, ambientazione notturna invernale, render 3D
        cinematografico. Stessa composizione centrata sulla toilette sporca, ma ora il liquido nel water
        è parzialmente ghiacciato in superficie, con crepe nel ghiaccio rossastro-scuro sottostante.
        Bordo del water incrostato di brina, colature scure ora semi-solidificate dal freddo.

        Dietro, la stessa grande finestra multipla ma con diversi pannelli di vetro rotti da cui entra
        una tormenta di neve leggera, fiocchi visibili controluce, cielo notturno con luna piena filtrata
        da nuvole di neve. Luce lunare blu-ghiaccio ancora più fredda e tagliente come fonte principale.

        A sinistra lo specchio ovale incrinato, ora appannato da condensa ghiacciata ai bordi, sopra il
        lavabo vintage con rubinetti coperti di ghiaccioli, righe di sangue congelato lungo il muro.
        A destra la vasca vintage con piedi a zampa di leone, bordo coperto di neve accumulata, liquido
        scuro semi-gelato che cola lentamente.

        Pavimento con piastrelle rotte coperte da un sottile strato di neve soffiata dalla finestra,
        pozze di sangue ora ghiacciate e opache, frammenti di ossa che spuntano dalla neve. Carta da
        parati damascata, macchiata, ora anche rigata da gelo e condensa.

        Illuminazione: gelida, blu-bianca dominante, altissimo contrasto tra il bianco della neve e il
        rosso/ruggine scuro quasi nero del sangue congelato. Stessa architettura, stessa inquadratura
        grandangolare, cambia solo la stagione e la palette.
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto decadente sommersa nel gelo invernale, render 3D fotorealistico. La stessa
        lampadina nuda pende dal soffitto ma ora la sua luce rossa si mescola con un pallido bagliore
        blu-ghiaccio che filtra dalla finestra, creando una palette insolita di rosso freddo e blu
        gelido invece del rosso puro originale.

        Letto disfatto con lenzuola rosse ora rigide di brina agli angoli, bottiglie di vetro scuro
        vuote coperte da un velo di gelo, riviste e giornali induriti dal freddo sparsi sul letto e
        sul pavimento di legno, anch'esso ricoperto da un sottile strato di neve entrata dalla finestra
        rotta.

        A sinistra finestra con vetri scuri incrinati da cui filtra vento gelido e neve, tende pesanti
        rigide di ghiaccio, comò con posacenere il cui mozzicone è ormai spento e coperto di brina.
        A destra comodino con cassetto aperto e vuoto, posacenere pieno di mozziconi congelati, scaffale
        in legno coperto da un leggero strato di gelo.

        Pareti scrostate con crepe ora rivestite di brina bianca che le rende ancora più inquietanti,
        intonaco che si sgretola per il gelo.

        Illuminazione: contrasto insolito tra rosso della lampadina e blu gelido dell'ambiente, ombre
        profonde nero-bordeaux mescolate a riflessi ghiacciati. Stessa inquadratura dal basso verso il
        letto, stessa disposizione degli arredi.
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata sepolta da neve accumulata, ambientazione notturna invernale, render 3D
        fotorealistico. Stessa composizione simmetrica sulla finestra a doppio battente sul fondo, i
        cui vetri rotti lasciano entrare un vento carico di neve, creando un cono di luce blu-bianca
        ancora più intensa e fredda, con fiocchi di neve visibili come particelle nel fascio volumetrico
        al posto della polvere.

        Lavello vintage in ceramica bianca ora coperto da un cumulo di neve entrata dalla finestra,
        canovaccio giallo rigido e ghiacciato appeso al bordo. Fornello a gas bianco arrugginito,
        colature scure ora congelate lungo la superficie. Pensili scrostati con un leggero strato di
        brina sugli sportelli aperti.

        Pavimento coperto non solo dal solito disordine (bottiglie, lattine, mozziconi, vetri, giornali)
        ma anche da un tappeto di neve soffice che si è accumulato vicino alla finestra, con impronte
        poco distinguibili sulla superficie ghiacciata. La grande pozza in primo piano è ora parzialmente
        gelata, con crepe sulla sua superficie riflettente.

        Pareti scrostate e ammuffite ai lati, ora anche macchiate di gelo, che si dissolvono nel buio
        profondo ai margini dell'inquadratura.

        Illuminazione: identica struttura chiaroscurale ma con dominante ancora più fredda e cristallina,
        bianco-blu gelido che sostituisce parzialmente il grigio, stessa inquadratura frontale simmetrica.
        """

        #teatro
        PROMPT5 = """
        Antico teatro all'italiana invernale, stessa vista da palco/balconata laterale verso il sipario,
        render 3D fotorealistico. Il grande lucernario o le finestre laterali (immaginate fuori campo)
        lasciano intuire un esterno innevato tramite una luce bianco-azzurra fredda che sostituisce i
        toni caldi originali, mescolandosi ai dorati del teatro in un contrasto inedito.

        Stesso pavimento in parquet a intarsio geometrico, ora opacizzato da un velo di polvere di
        ghiaccio, lucido ma freddo al tatto visivo. Le due colonne laterali in legno scuro, con le
        stesse mensole in metallo nero e i ganci in ferro, ora leggermente brinati sui bordi superiori
        per l'umidità gelida che si è infiltrata nell'edificio.

        Il sipario di velluto rosso bordeaux, con le stesse frange dorate, appare più cupo sotto la
        luce fredda, quasi violaceo in certi punti, mentre la cornice scenica dorata riflette la luce
        blu-bianca creando bagliori insoliti.

        Balaustra in legno tornito identica, platea sottostante con le poltrone rosse ora percepite più
        scure, palchi laterali nell'ombra con un accenno di brina sulle balaustre dorate.

        Illuminazione: contrasto freddo-caldo insolito, base gelida blu-bianca che invade i toni dorati
        e bordeaux originali, stessa prospettiva centrale simmetrica verso il sipario.
        """

        #portone
        PROMPT6 = """
        Grande portone metallico a due ante, ora ricoperto da uno spesso strato di ghiaccio e neve
        accumulata, su sfondo completamente buio, render 3D fotorealistico. Il metallo arrugginito è
        parzialmente coperto da cristalli di ghiaccio e ghiaccioli che pendono dai bordi dei pannelli
        rivettati, con la ruggine bruno-rossastra visibile a chiazze sotto lo strato ghiacciato.

        La sbarra/trave metallica orizzontale centrale è incrostata di brina, gli anelli metallici alle
        estremità coperti di ghiaccio sottile. Le catene che sostengono la bilancia a doppio piatto sono
        rigide e ghiacciate, i due piatti metallici semi-coperti da neve accumulata, leggermente
        sbilanciati come nell'originale.

        Sullo sfondo in alto, lo stesso piccolo cono di luce bianca fredda ora accompagnato da fiocchi
        di neve visibili nel fascio volumetrico invece della semplice polvere, mentre il resto rimane
        immerso nell'oscurità. Ragnatele agli angoli ora ricoperte di brina sottile, quasi cristallizzate.

        Pavimento in pietra/cemento grezzo coperto da un leggero strato di neve accumulata sotto le
        bilance. Pareti laterali con crepe nell'intonaco ora rivestite da gelo.

        Illuminazione: altissimo contrasto identico all'originale ma con dominante ancora più glaciale,
        bianco-blu intenso in alto contro il nero assoluto, stessa inquadratura frontale e simmetrica.
        """

    elif Livel == 3:
        # stile solare, pacifico ma inquietante — luce calda e dorata sovraesposta, horror "in pieno giorno"
        #attrezzi
        PROMPT1 = """
        La stessa officina/sala chirurgica improvvisata in stile horror, ma ora invasa da una luce
        solare calda e dorata che filtra da una finestrella alta sul soffitto (assente nell'originale
        ma qui presente), creando un contrasto inquietante tra l'ambientazione macabra e una luce quasi
        idilliaca, quasi da pomeriggio d'estate. Pareti in cemento scrostato ora tinte di ocra dorata,
        con pulviscolo dorato sospeso nell'aria invece della polvere grigia.

        Stesso pannello perforato con gli stessi utensili arrugginiti, ora illuminati con riflessi caldi
        dorati che rendono le lame quasi luccicanti nonostante la ruggine.

        Stesso banco di lavoro graffiato, sangue secco che ora appare bruno-ruggine sotto la luce calda
        invece che nero, stesso tronchese, stesso straccio, stesso bisturi, stessa disposizione esatta.

        Atmosfera: paradosso inquietante tra la luce calda, dorata, quasi rassicurante e gli strumenti
        macabri sul tavolo — l'horror "alla luce del sole" risulta ancora più perturbante per il
        contrasto. Fotorealistico, cinematografico, palette calda satura (ambra, oro, ocra) con ombre
        morbide invece che dure.

        Camera: identica inquadratura centrata e simmetrica verso il banco e la parete degli attrezzi.
        Mantieni la stessa disposizione di oggetti, cambia solo luce e palette.
        """

        #bagno
        PROMPT2 = """
        Bagno abbandonato bagnato da una luce solare calda e intensa che filtra dalla grande finestra
        multipla sul fondo, i cui pannelli rotti lasciano intravedere un cielo diurno terso, azzurro
        intenso, invece della notte con luna piena. Render 3D cinematografico, stessa composizione
        centrata sulla toilette sporca.

        Il liquido nel water resta rossastro scuro ma ora riflette bagliori dorati sulla superficie
        increspata, la sostanza scura che cola sul pavimento appare quasi calda sotto questa luce.

        A sinistra lo stesso specchio ovale incrinato, ora con riflessi dorati distorti, sopra il
        lavabo vintage rigato di sangue che appare quasi ambrato controluce. A destra la vasca vintage
        con piedi a zampa di leone, illuminata da un fascio caldo che ne accentua le texture vintage.

        Pavimento con le stesse piastrelle rotte, pozze di sangue secco e liquido nero viscoso, ma ora
        baciate da una luce dorata che crea riflessi caldi sulla superficie bagnata. Carta da parati
        damascata sbiadita, ora con tonalità calde accentuate dal sole.

        Illuminazione: calda, dorata, satura, in netto contrasto con l'inquietudine della scena — il
        sole non rassicura ma amplifica il disagio. Stessa inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto decadente illuminata da una calda luce solare pomeridiana che entra dalla
        finestra a sinistra, sostituendo la lampadina rossa come fonte di luce dominante (la lampadina
        resta accesa ma appare debole contro la luce naturale). Render 3D fotorealistico, palette calda
        dorata-ambrata che si mescola ai toni rossi residui creando un'atmosfera stranamente calma ma
        inquietante.

        Stesso letto disfatto con lenzuola rosse, bottiglie di vetro scuro vuote che ora luccicano sotto
        il sole, riviste e giornali sparsi illuminati caldamente. Pavimento di legno con le stesse
        bottiglie rovesciate e pagine sparse, ora visibili chiaramente sotto la luce diurna.

        A sinistra finestra con tende pesanti scostate che lasciano entrare fasci di luce dorata polverosa,
        comò con posacenere acceso il cui fumo si arrotola visibile nel controluce. A destra comodino con
        cassetto aperto, posacenere pieno, scaffale in legno, tutti baciati dalla stessa luce calda.

        Pareti scrostate con crepe profonde, ora ben visibili e quasi "belle" sotto la luce solare,
        intonaco scrostato dorato.

        Illuminazione: calda, dorata, diffusa, contrasto morbido invece che duro — l'horror psicologico
        persiste ma diventa più sottile, quasi malinconico. Stessa inquadratura dal basso verso il letto.
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata invasa da una luce solare intensa e calda che entra dalla finestra a doppio
        battente sul fondo, i cui vetri sporchi filtrano un pomeriggio dorato invece della notte fredda.
        Render 3D fotorealistico, stessa composizione simmetrica centrata sulla finestra.

        Lavello vintage in ceramica bianca pieno di piatti sporchi, ora illuminato caldamente, canovaccio
        giallo che risalta ancora di più sotto il sole. Fornello a gas bianco arrugginito con colature
        scure che appaiono bruno-calde. Pensili scrostati con sportelli aperti, illuminati da riflessi
        dorati.

        Pavimento con lo stesso disordine di bottiglie, lattine, mozziconi, vetri rotti, giornali sparsi,
        ora tutto ben visibile e quasi "pittoresco" sotto la luce intensa. La grande pozza in primo piano
        riflette il cielo dorato invece della debole luce blu-grigia.

        Pareti scrostate e ammuffite ai lati, ora rivelate chiaramente dalla luce diurna invece di
        dissolversi nel buio, aumentando paradossalmente il senso di squallore visibile.

        Illuminazione: calda, dorata, quasi accecante in certi punti, contrasto ridotto rispetto
        all'originale ma disagio amplificato dalla piena visibilità del degrado. Stessa inquadratura
        frontale simmetrica.
        """

        #teatro
        PROMPT5 = """
        Antico teatro all'italiana bagnato da una calda luce solare che filtra da un lucernario immaginario
        sopra il sipario, sostituendo la penombra originale con un pomeriggio dorato e polveroso. Render
        3D fotorealistico, stessa vista da palco/balconata laterale.

        Stesso pavimento in parquet a intarsio geometrico, ora lucidissimo e dorato sotto la luce solare,
        riflessi caldi diffusi invece del singolo punto luminoso. Colonne laterali in legno scuro con le
        stesse mensole e ganci in ferro, ora baciate da fasci di luce che attraversano l'aria polverosa.

        Sipario di velluto rosso bordeaux con frange dorate, ora acceso da riflessi solari che lo rendono
        quasi vivido, cornice scenica dorata scintillante sotto il sole.

        Balaustra in legno tornito, platea con poltrone rosse ora ben illuminate, palchi laterali che
        escono dall'ombra rivelando dettagli dorati altrimenti nascosti.

        Illuminazione: calda, dorata, diffusa e quasi teatrale essa stessa, contrasto morbido, atmosfera
        solenne ma stranamente serena — l'inquietudine nasce dal vuoto e dal silenzio più che dall'ombra.
        Stessa prospettiva centrale simmetrica verso il sipario.
        """

        #portone
        PROMPT6 = """
        Grande portone metallico a due ante, ora illuminato da un intenso raggio di luce solare calda
        che scende dall'alto invece del freddo cono bianco originale, su sfondo scuro ma meno assoluto.
        Render 3D fotorealistico.

        Il metallo arrugginito riflette calde tonalità ambrate sotto il sole, i pannelli rivettati e le
        colature verticali simili a sangue secco appaiono bruno-dorate invece che nere. Stessa sbarra
        metallica orizzontale, stessi anelli, stesse catene che sostengono la bilancia a doppio piatto,
        ora scintillanti di riflessi caldi.

        Il cono di luce dall'alto è ora dorato e intenso, con pulviscolo dorato sospeso invece di polvere
        fredda, mentre il resto della scena resta in ombra profonda per contrasto. Ragnatele agli angoli
        che catturano la luce calda creando riflessi dorati sottili.

        Pavimento in pietra/cemento grezzo, illuminato parzialmente da riflessi caldi. Pareti laterali
        con crepe nell'intonaco scuro, appena percettibili nella penombra residua.

        Illuminazione: altissimo contrasto mantenuto ma con dominante calda-dorata al posto del
        bianco-freddo, stessa inquadratura frontale e perfettamente simmetrica.
        """

    elif Livel == 4:
        # stile ufficio/palazzo — horror corporate, luci al neon/fluorescenti, grigio-beige-verde acido
        #attrezzi
        PROMPT1 = """
        Una sala riunioni/ufficio dismesso trasformato in un'improvvisata sala degli strumenti in stile
        horror corporate, illuminata da un unico tubo fluorescente al soffitto che sfarfalla debolmente,
        proiettando una luce verdastra-biancastra fredda. Pareti in cartongesso scrostato con moquette
        grigia consumata sul pavimento, macchie di umidità sui pannelli del controsoffitto.

        Sulla parete di fondo, al posto del pannello perforato, uno scaffale metallico da ufficio (tipo
        scaffalatura industriale) mostra la stessa varietà di utensili e lame arrugginiti — seghetto,
        bisturi, pinze, forbici, trapano manuale, coltelli — disposti tra faldoni e raccoglitori polverosi.

        In primo piano, una scrivania metallica graffiata sostituisce il banco di lavoro, coperta dalle
        stesse macchie di sangue secco e chiodi arrugginiti, con lo stesso tronchese/tagliaossa al
        centro-sinistra, straccio sporco a destra, bisturi in basso a destra, tra fogli di documenti
        sparsi e una tastiera rotta.

        Atmosfera: cupa, asettica e inquietante, luce fluorescente fredda che sfarfalla, contrasto tra
        l'ambiente "normale" da ufficio e gli strumenti macabri, estetica horror corporate, fotorealistico,
        cinematografico, texture dettagliate su metallo, moquette e cartongesso.

        Camera: identica inquadratura centrata e simmetrica verso la scrivania e lo scaffale.
        """

        #bagno
        PROMPT2 = """
        Bagno di un ufficio abbandonato, ambientazione notturna, render 3D cinematografico. Stessa
        composizione centrata su una toilette da bagno pubblico/aziendale in ceramica bianca sporca,
        coperchio alzato, water riempito di liquido rossastro scuro, con la stessa sostanza scura che
        cola sul pavimento in piastrelle da ufficio.

        Dietro, al posto della finestra a vetri multipli, una finestra rettangolare con vetro smerigliato
        da ufficio, parzialmente rotta, da cui filtra la luce fredda e sfarfallante di un cartellone
        pubblicitario/insegna esterna, in tonalità blu-verdastre invece della luce lunare.

        A sinistra lo stesso specchio ora rettangolare da bagno aziendale, incrinato, sopra un lavabo
        in acciaio inox da ufficio, rigato di sangue. A destra, al posto della vasca, una fila di
        cabine dei bagni con porte scardinate, semi in ombra, con liquido scuro che cola da sotto una
        delle porte.

        Pavimento con piastrelle industriali rotte, pozze di sangue secco e liquido nero viscoso, resti
        di documenti aziendali bagnati sparsi tra le pozze. Pareti piastrellate bianche, macchiate di
        schizzi di sangue e muffa da umidità.

        Illuminazione: fredda, fluorescente, sfarfallante, palette grigio-verde-blu che contrasta con il
        rosso del sangue. Stessa inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Un ufficio dirigenziale/camera privata di un palazzo aziendale abbandonato, trasformato in una
        stanza decadente in stile horror, illuminata da un'unica plafoniera al neon rosso guasta che
        sfarfalla dal soffitto, sostituendo la lampadina nuda ma mantenendo la stessa dominante rossa.

        Al posto del letto, un divano/chaise longue da ufficio disfatto con cuscini rossi, sul quale sono
        sparse le stesse bottiglie di vetro scuro vuote, riviste aziendali e giornali aperti. Il pavimento
        in moquette rossa consumata è coperto da altre bottiglie rovesciate e documenti sparsi, suggerendo
        abbandono e disagio.

        A sinistra una vetrata da ufficio con persiane rotte e vetri scuri, e una scrivania in legno con
        un piccolo posacenere acceso. A destra un mobile archivio con un cassetto aperto e vuoto,
        sormontato da un posacenere pieno, e dietro una libreria/scaffale con faldoni e un cassetto
        leggermente aperto.

        Pareti in cartongesso rovinato con crepe profonde che si diramano, pannelli del controsoffitto
        caduti, cavi elettrici a vista.

        Illuminazione: dominante rossa dal neon guasto, sfarfallio intermittente, forti contrasti tra zone
        rosse vivide e ombre nere. Stessa inquadratura prospettica dal basso verso il divano.
        """

        #cucina
        PROMPT4 = """
        Cucina/area break di un ufficio abbandonato, ambientazione notturna molto buia, render 3D
        fotorealistico in stile horror corporate. Composizione simmetrica centrata su una grande vetrata
        a doppio battente sul fondo (invece della finestra domestica), con vetri sporchi da cui filtra
        la debole luce blu-verdastra di un'insegna al neon esterna, unica fonte di illuminazione.

        Al centro, sotto la vetrata, un lavello industriale in acciaio inox pieno di tazze e piatti
        sporchi, con un canovaccio giallo appeso al bordo. A destra, un frigorifero da ufficio macchiato
        e arrugginito con lo sportello socchiuso, colature scure lungo la superficie. Sopra il lavello,
        pensili da cucina office scrostati con sportelli aperti, alcuni con bottiglie di vetro scuro
        disposte sopra.

        Il pavimento in primo piano è ricoperto di disordine: bottiglie di vetro scuro vuote, lattine
        di bibite schiacciate, mozziconi di sigaretta, bicchieri di plastica rotti, tazze rotte, e
        documenti/fogli aziendali sparpagliati. Al centro, una grande pozza di liquido che riflette
        debolmente la luce dell'insegna, sopra fogli bagnati e macchiati.

        Pareti scrostate ai lati, immerse quasi completamente nell'oscurità.

        Illuminazione: estremamente scura, dominata da ombre nere profonde, unica fonte blu-verdastra
        fredda dall'insegna esterna. Stessa inquadratura frontale e simmetrica.
        """

        #teatro (qui reinterpretato come auditorium/sala conferenze aziendale)
        PROMPT5 = """
        Interno di un vecchio auditorium/sala conferenze aziendale in stile anni '80, vista da una
        balconata laterale verso il palco centrale, render 3D fotorealistico, atmosfera cupa e
        misteriosa. In primo piano, pavimento in moquette grigia consumata con motivo geometrico
        sbiadito, illuminato da un fascio di luce soffusa proveniente da un proiettore rotto che crea
        un punto luminoso tremolante sul pavimento.

        Ai lati, due grandi pannelli acustici in legno scuro, con basamenti in metallo e ganci per cavi,
        sormontati da altoparlanti neri fuori uso. I pannelli incorniciano la scena in composizione
        simmetrica.

        Sullo sfondo centrale, un grande schermo di proiezione bianco-grigio strappato, incorniciato da
        tende pesanti blu scuro bordate di frange consumate, racchiuso da una struttura metallica
        scenica. Davanti allo schermo, un piccolo podio rialzato con un leggio abbandonato.

        Tra il primo piano e il fondale, una balaustra metallica che separa la balconata dalla platea
        sottostante, dove si intravedono file di sedie da conferenza grigie. Ai lati, palchi laterali
        con cabine di regia buie.

        Illuminazione: penombra con toni freddi grigio-blu, luce drammatica e soffusa dal proiettore
        rotto, forte contrasto tra ombre e il punto luminoso tremolante. Stessa prospettiva centrale
        simmetrica.
        """

        #portone
        PROMPT6 = """
        Scena horror corporate con una grande porta blindata a due ante in primo piano (ingresso di un
        caveau/archivio aziendale), su sfondo completamente buio, render 3D fotorealistico. La porta è
        in metallo industriale arrugginito, con pannelli rettangolari rivettati e un badge/lettore
        elettronico rotto al centro, texture di ruggine bruno-rossastra con colature verticali simili a
        sangue secco.

        Al centro della porta, una grande sbarra/trave metallica di sicurezza orizzontale funge da
        blocco manuale d'emergenza, fissata con anelli metallici. Da questi anelli pendono catene che
        sostengono, al posto della bilancia, due grandi faldoni/scatole di documenti metalliche
        sospese, leggermente sbilanciate, che creano un elemento simbolico inquietante.

        Sullo sfondo, in alto, un piccolo cono di luce bianca fredda da un faro di emergenza illumina
        debolmente la parte superiore, con particelle di polvere sospese, mentre il resto rimane
        nell'oscurità. Cavi elettrici scoperti pendono dagli angoli come ragnatele moderne.

        Il pavimento in primo piano è in cemento industriale grezzo. Le pareti laterali mostrano crepe
        profonde nel cemento armato a vista.

        Illuminazione: altissimo contrasto, buio quasi assoluto interrotto da luce bianco-fredda
        dall'alto, contrasto coi toni ruggine della porta blindata. Stessa inquadratura frontale e
        perfettamente simmetrica.
        """

    elif Livel == 5:
        # inferno fuoco — palette rosso/arancio/nero, brace, fumo, calore
        #attrezzi
        PROMPT1 = """
        Officina/sala chirurgica improvvisata immersa in un bagliore infernale, stile horror, illuminata
        non più da una lampadina ma da un bagliore rosso-arancio pulsante che sembra provenire da crepe
        nel pavimento o dalle pareti stesse, come riflessi di fuoco lontano. Pareti in cemento scrostato
        annerite dalla fuliggine, con crepe incandescenti che si diramano come vene di lava.

        Sulla parete di fondo, lo stesso pannello perforato in legno, ora parzialmente carbonizzato, mostra
        gli stessi utensili e lame arrugginiti, ora anneriti dal fumo e con riflessi arancioni sulle
        superfici metalliche.

        Il banco di lavoro in legno graffiato mostra segni di bruciature ai bordi, coperto dalle stesse
        macchie di sangue secco che ora appaiono quasi fuse col bagliore rosso, stesso tronchese al
        centro-sinistra, stesso straccio a destra, stesso bisturi in basso a destra, cenere sottile sparsa
        sulla superficie.

        Atmosfera: infernale, opprimente, calda, con particelle di cenere e brace sospese nell'aria simili
        a lucciole, forte contrasto tra il bagliore arancione e le ombre nere fuligginose. Fotorealistico,
        cinematografico, texture bruciate e carbonizzate ultra-dettagliate.

        Camera: identica inquadratura centrata e simmetrica verso il banco e la parete degli attrezzi.
        """

        #bagno
        PROMPT2 = """
        Bagno abbandonato avvolto da un bagliore infernale, ambientazione cupa, render 3D cinematografico.
        Stessa composizione centrata sulla toilette sporca, ma il liquido nel water ora ribolle
        leggermente, quasi fosse lava scura mista a sangue, con vapore rosso-arancio che si solleva dalla
        superficie.

        Dietro, la stessa finestra a vetri multipli rotti, ma invece del cielo notturno rivela un bagliore
        arancione infernale, come se all'esterno bruciasse tutto, con sagome di rami d'albero anneriti e
        contorti in controluce. Luce rosso-arancio calda e pulsante come fonte principale al posto della
        luce lunare fredda.

        A sinistra lo specchio ovale incrinato riflette bagliori di fuoco distorti, sopra il lavabo vintage
        rigato di sangue che ora sembra fondersi col calore ambientale. A destra la vasca vintage con
        liquido denso e scuro che ora fuma leggermente.

        Pavimento con piastrelle rotte annerite dalla fuliggine, pozze di sangue secco e liquido nero
        viscoso che riflettono bagliori arancioni. Carta da parati vintage bruciacchiata ai bordi, macchiata
        di sangue e fuliggine.

        Illuminazione: calda, satura, dominante rosso-arancio infernale, altissimo contrasto con ombre
        nere fuligginose. Stessa inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto immersa in un bagliore d'inferno, render 3D fotorealistico. La lampadina nuda rossa
        pende ancora dal soffitto ma ora è affiancata da un bagliore pulsante arancione proveniente da
        crepe nel muro, come se dietro la parete ardesse un fuoco perenne, intensificando la dominante
        rossa originale in una palette rosso-arancio-nero fumo.

        Letto disfatto con lenzuola rosse ora leggermente bruciacchiate ai bordi, bottiglie di vetro scuro
        vuote, riviste e giornali sparsi, alcuni con angoli anneriti come se stessero per prendere fuoco.
        Pavimento di legno annerito dalla fuliggine, coperto da altre bottiglie rovesciate e pagine sparse.

        A sinistra finestra con tende pesanti che lasciano intravedere un bagliore arancione esterno,
        comò con un posacenere il cui mozzicone ora sembra una vera brace ardente. A destra comodino con
        cassetto aperto, posacenere pieno di braci accese, scaffale in legno con superficie carbonizzata.

        Pareti scrostate con crepe che ora sembrano vene incandescenti, intonaco annerito dal fumo.

        Illuminazione: dominante rosso-arancio intensa e pulsante, ombre nero-fuligginose profonde,
        atmosfera opprimente e calda. Stessa inquadratura dal basso verso il letto.
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata e devastata da un incendio latente, ambientazione molto buia rischiarata da
        bagliori di brace, render 3D fotorealistico in stile horror infernale. Composizione simmetrica
        centrata sulla finestra a doppio battente sul fondo, i cui vetri anneriti lasciano filtrare un
        bagliore arancione pulsante invece della luce lunare, come se all'esterno ardesse un fuoco
        costante.

        Lavello vintage in ceramica ora annerita dal fumo, pieno di piatti sporchi, canovaccio giallo
        bruciacchiato appeso al bordo. Fornello a gas bianco con colature scure che ora sembrano quasi
        fuse dal calore, fiamme fantasma che tremolano appena visibili nei fornelli. Pensili scrostati con
        superfici carbonizzate.

        Pavimento coperto dal solito disordine — bottiglie, lattine, mozziconi, vetri rotti, giornali —
        ma ora con cenere sottile sparsa ovunque e piccole braci che luccicano tra i detriti. La grande
        pozza in primo piano riflette il bagliore arancione della finestra invece della luce blu-grigia.

        Pareti scrostate e annerite ai lati, che si dissolvono nell'oscurità fumosa ai margini
        dell'inquadratura.

        Illuminazione: scura e cupa ma pulsante di bagliori arancioni, forte contrasto chiaroscurale,
        particelle di cenere sospese nel fascio di luce al posto della polvere. Stessa inquadratura
        frontale simmetrica.
        """

        #teatro
        PROMPT5 = """
        Antico teatro all'italiana avvolto da un bagliore infernale, stessa vista da palco/balconata
        laterale verso il sipario, render 3D fotorealistico. Il sipario di velluto rosso bordeaux appare
        ora quasi in fiamme nella tonalità, illuminato da un bagliore arancione pulsante che sembra
        provenire da dietro il sipario stesso, come se qualcosa ardesse sul palco nascosto.

        Stesso pavimento in parquet a intarsio geometrico, ora con riflessi caldi arancioni al posto del
        singolo punto luminoso freddo, leggero velo di cenere sulla superficie lucida. Le due colonne
        laterali in legno scuro mostrano segni di bruciature superficiali, mensole in metallo nero
        arroventate ai bordi.

        Cornice scenica dorata che ora riflette bagliori infuocati invece della luce calda dorata neutra,
        frange dorate del sipario che sembrano quasi luccicare come braci.

        Balaustra in legno tornito, platea con poltrone rosse che sembrano fondersi nel bagliore, palchi
        laterali semi-visibili tra fumo sottile che aleggia nell'aria.

        Illuminazione: dominante rosso-arancio pulsante, altissimo contrasto tra ombre fuligginose e
        bagliore infernale, atmosfera opprimente e minacciosa invece che solenne. Stessa prospettiva
        centrale simmetrica.
        """

        #portone
        PROMPT6 = """
        Grande portone metallico a due ante immerso in un bagliore infernale, su sfondo completamente
        scuro rischiarato da un pulsare arancione, render 3D fotorealistico. Il metallo arrugginito è
        reso incandescente in alcuni punti, come se fosse riscaldato dall'interno, con crepe luminose
        color brace che attraversano i pannelli rivettati al posto delle semplici colature di ruggine.

        Al centro, la stessa sbarra/trave metallica orizzontale ora arroventata, fissata con anelli
        metallici che brillano di calore residuo. Le catene che sostengono la bilancia a doppio piatto
        sono anch'esse rovent, i due piatti metallici che sembrano contenere brace ardente al posto del
        semplice sbilanciamento simbolico originale.

        Sullo sfondo, in alto, al posto del cono di luce bianca fredda, un bagliore arancione pulsante
        che sembra provenire da un fuoco lontano oltre il portone, con particelle di cenere sospese in
        aria invece della polvere. Ragnatele agli angoli parzialmente bruciacchiate.

        Il pavimento in pietra/cemento grezzo mostra crepe incandescenti sottili. Le pareti laterali
        mostrano crepe profonde nell'intonaco scuro, alcune che pulsano di un bagliore residuo.

        Illuminazione: altissimo contrasto, nero fuligginoso profondo interrotto da bagliori arancio-rossi
        pulsanti, atmosfera infernale e minacciosa. Stessa inquadratura frontale e perfettamente
        simmetrica.
        """

    elif Livel == 6:
        # giungla/vegetazione abbandonata — verde muschio, umidità, marciume vegetale
        #attrezzi
        PROMPT1 = """
        Officina/sala chirurgica improvvisata invasa dalla vegetazione, stile horror, illuminata da una
        singola lampadina nuda che pende dal soffitto, la cui luce filtra tra radici e liane che sono
        penetrate attraverso crepe nel soffitto di cemento. Pareti in cemento scrostato ricoperte di
        muschio verde scuro e macchie di umidità, con radici che si insinuano tra i mattoni visibili.

        Sulla parete di fondo, il pannello perforato in legno mostra gli stessi utensili e lame
        arrugginiti, ora parzialmente avvolti da liane sottili e coperti da una patina di muffa verdastra.

        Il banco di lavoro in legno graffiato è invaso da funghi e muschio ai bordi, coperto dalle stesse
        macchie di sangue secco ormai mescolate a chiazze di marciume vegetale verde-nero, stesso
        tronchese al centro-sinistra, stesso straccio a destra, stesso bisturi in basso a destra, tra
        foglie marce cadute dal soffitto.

        Atmosfera: umida, soffocante, verde-marrone desaturato, gocciolio d'acqua visibile, forte contrasto
        tra la luce gialla della lampadina e le ombre verdi-nere profonde. Fotorealistico, cinematografico,
        texture di muschio, radici e legno marcio ultra-dettagliate.

        Camera: identica inquadratura centrata e simmetrica verso il banco e la parete degli attrezzi.
        """

        #bagno
        PROMPT2 = """
        Bagno abbandonato invaso dalla vegetazione, ambientazione notturna, render 3D cinematografico.
        Stessa composizione centrata sulla toilette sporca, ora con radici che escono dalle crepe del
        water stesso, muschio che cresce sul bordo di ceramica, liquido rossastro scuro nel water ora
        velato da foglie marce galleggianti.

        Dietro, la stessa grande finestra a vetri multipli rotti, ma ora invasa da rampicanti e liane che
        si intrecciano attraverso i vetri rotti, rivelando uno squarcio di foresta buia con sagome di
        alberi contorti al posto dei rami spogli. Luce lunare verde-bluastra filtrata dal fogliame come
        fonte principale.

        A sinistra lo specchio ovale incrinato, ora parzialmente coperto da muschio ai bordi, sopra il
        lavabo vintage rigato di sangue e umidità verdastra. A destra la vasca vintage con radici che
        fuoriescono dallo scarico, liquido scuro misto a foglie marce che cola lungo il fianco.

        Pavimento con piastrelle rotte invase da muschio e piccole piante infestanti che crescono dalle
        crepe, pozze di sangue secco mescolate a fango vegetale. Carta da parati vintage marcita e
        rigonfia per l'umidità, macchiata di muffa verde oltre che di sangue.

        Illuminazione: cupa, umida, dominante verde-bluastra desaturata in contrasto coi toni rosso-ruggine
        del sangue. Stessa inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto decadente invasa dalla natura, render 3D fotorealistico. La lampadina rossa nuda
        pende ancora dal soffitto ma la sua luce ora filtra attraverso un intrico di liane pendenti dal
        soffitto crepato, creando ombre organiche e mobili sulle pareti, mantenendo la dominante rossa
        ma mescolata a riflessi verde scuro.

        Letto disfatto con lenzuola rosse ora parzialmente ricoperte da muschio e piccole radici emerse
        dal materasso stesso, bottiglie di vetro scuro vuote, riviste e giornali marciti sparsi. Pavimento
        di legno rigonfio e marcito per l'umidità, coperto da altre bottiglie rovesciate e vegetazione
        infestante che cresce tra le assi.

        A sinistra finestra con tende pesanti in decomposizione, vetri scuri coperti da rampicanti esterni,
        comò in legno tarlato con un piccolo posacenere e un germoglio che cresce accanto. A destra
        comodino con cassetto aperto invaso da muffa, posacenere pieno, scaffale in legno con funghi che
        crescono sui ripiani.

        Pareti scrostate con crepe profonde da cui filtrano radici sottili, intonaco scrostato e coperto
        di muschio.

        Illuminazione: dominante rossa filtrata attraverso il fogliame, ombre verdi-nere organiche e
        mobili, atmosfera claustrofobica e soffocante. Stessa inquadratura dal basso verso il letto.
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata e invasa dalla vegetazione, ambientazione notturna molto buia, render 3D
        fotorealistico in stile horror/abbandono. Composizione simmetrica centrata sulla finestra a
        doppio battente sul fondo, ora quasi completamente coperta da rampicanti esterni, con vetri
        sporchi e foglie premute contro il vetro da cui filtra una debole luce verde-bluastra.

        Lavello vintage in ceramica bianca invaso da muschio, pieno di piatti sporchi coperti di muffa,
        canovaccio giallo marcito appeso al bordo. Fornello a gas bianco arrugginito con radici che
        fuoriescono dalle fessure, colature scure miste a umidità vegetale. Pensili scrostati con
        sportelli aperti, alcuni con piccole piante infestanti cresciute all'interno.

        Il pavimento in primo piano è ricoperto dal solito disordine — bottiglie, lattine, mozziconi,
        vetri rotti, giornali — ora inzuppati e ricoperti da uno strato di foglie marce e muschio. La
        grande pozza in primo piano è ora un piccolo stagno melmoso con foglie galleggianti.

        Pareti scrostate e ammuffite ai lati, invase da radici che si diramano nell'oscurità profonda ai
        margini dell'inquadratura.

        Illuminazione: scura e cupa, dominante verde-bluastra filtrata dal fogliame esterno, forte
        contrasto chiaroscurale. Stessa inquadratura frontale simmetrica.
        """

        #teatro
        PROMPT5 = """
        Antico teatro all'italiana invaso dalla natura dopo decenni di abbandono, stessa vista da
        palco/balconata laterale verso il sipario, render 3D fotorealistico. Crepe nel soffitto lasciano
        filtrare radici e liane che pendono verso il palco, mescolando l'eleganza decadente originale a
        un senso di natura che reclama lo spazio.

        Stesso pavimento in parquet a intarsio geometrico, ora rigonfio e scheggiato per l'umidità, con
        muschio che cresce nelle fughe tra i pannelli, ancora lucido a tratti dove la luce lo raggiunge.
        Le due colonne laterali in legno scuro sono avvolte da rampicanti sottili, mensole in metallo
        nero ossidate dal verde ramino.

        Il sipario di velluto rosso bordeaux appare marcito e strappato in alcuni punti, con muffa verde
        che si diffonde dai bordi verso il centro, cornice scenica dorata ossidata e scrostata.

        Balaustra in legno tornito invasa da edera, platea sottostante con poltrone rosse coperte di
        muschio, palchi laterali semi-inghiottiti dalla vegetazione nell'ombra.

        Illuminazione: penombra con toni verde-marroni, luce drammatica filtrata dal fogliame che crea
        ombre mobili, atmosfera solenne ma ormai reclamata dalla natura, inquietante e malinconica. Stessa
        prospettiva centrale simmetrica.
        """

        #portone
        PROMPT6 = """
        Grande portone metallico a due ante, ora quasi sommerso da rampicanti ed edera che si arrampicano
        sui pannelli rivettati, su sfondo completamente buio, render 3D fotorealistico. Il metallo
        arrugginito è coperto in gran parte da vegetazione, con la ruggine bruno-rossastra visibile solo
        a chiazze tra le foglie, colature verticali simili a sangue secco che si mescolano a muschio umido.

        Al centro, la stessa sbarra/trave metallica orizzontale ora avvolta da liane, fissata con anelli
        metallici ossidati e coperti di verderame. Le catene che sostengono la bilancia a doppio piatto
        sono invase da rampicanti sottili, i due piatti metallici parzialmente riempiti di foglie marce
        e terra, ancora leggermente sbilanciati.

        Sullo sfondo, in alto, lo stesso piccolo cono di luce bianca fredda e polverosa, ora filtrato
        attraverso un intrico di rami e foglie che pendono dall'alto, creando ombre frastagliate nel
        fascio di luce. Ragnatele fitte agli angoli, mescolate a radici sottili che scendono dal soffitto.

        Il pavimento in pietra/cemento grezzo è invaso da muschio e piccole crepe da cui spuntano piante
        infestanti. Le pareti laterali mostrano crepe profonde da cui filtrano radici.

        Illuminazione: altissimo contrasto, buio quasi assoluto interrotto da luce fredda dall'alto
        filtrata dal fogliame, dominante verde-marrone che si aggiunge ai toni ruggine. Stessa
        inquadratura frontale e perfettamente simmetrica.
        """

    elif Livel == 7:
        # spazio/tecnologico — horror sci-fi, neon blu-viola, metallo, schermi
        #attrezzi
        PROMPT1 = """
        Officina/sala chirurgica improvvisata all'interno di una stazione spaziale abbandonata, stile
        horror sci-fi, illuminata da una singola striscia LED al soffitto che sfarfalla in un blu-viola
        freddo, sostituendo la lampadina a incandescenza. Pareti in pannelli metallici scrostati con
        condotti a vista, macchie di condensa e corrosione elettronica, piccoli schermi rotti che
        lampeggiano staticamente sulle pareti laterali.

        Sulla parete di fondo, al posto del pannello perforato in legno, un supporto metallico modulare
        mostra la stessa varietà di utensili e lame arrugginiti — seghetto, bisturi, pinze, forbici,
        trapano, coltelli — ora affiancati da strumenti chirurgici robotici spenti e inattivi.

        In primo piano, un banco di lavoro in metallo scanalato sostituisce quello in legno, coperto dalle
        stesse macchie di sangue secco e detriti metallici, stesso tronchese al centro-sinistra, stesso
        straccio a destra, stesso bisturi in basso a destra, cavi elettrici scoperti che sfrigolano
        debolmente.

        Atmosfera: fredda, tecnologica, inquietante, luce blu-viola pulsante intermittente, forte contrasto
        tra i bagliori dei LED e le ombre metalliche profonde. Fotorealistico, cinematografico, texture
        metalliche e tecnologiche ultra-dettagliate.

        Camera: identica inquadratura centrata e simmetrica verso il banco e la parete degli strumenti.
        """

        #bagno
        PROMPT2 = """
        Bagno/modulo igienico di una stazione spaziale abbandonata, ambientazione cupa, render 3D
        cinematografico. Stessa composizione centrata su un servizio igienico modulare in metallo
        sporco, coperchio alzato, riempito di liquido rossastro scuro che fluttua leggermente in modo
        innaturale, con la stessa sostanza scura che cola sul pavimento metallico.

        Dietro, al posto della finestra tradizionale, un oblò panoramico incrinato che rivela lo spazio
        profondo esterno, stelle lontane e il bagliore blu di un pianeta, con crepe che attraversano il
        vetro spesso. Luce blu-violacea fredda dallo spazio come fonte principale, mescolata a un
        residuo bagliore rosso di emergenza lampeggiante.

        A sinistra uno specchio digitale incrinato con superficie a schermo che mostra interferenze
        distorte, sopra un lavabo modulare in metallo rigato di sangue. A destra una cabina doccia
        modulare con pannelli scorrevoli semiaperti, liquido denso e scuro che si accumula sul fondo.

        Pavimento in grigliato metallico rotto, pozze di sangue secco e liquido nero viscoso che
        riflettono i bagliori blu, piccoli frammenti metallici sparsi. Pareti in pannelli tecnologici
        scrostati, macchiate di sangue e corrosione da esposizione.

        Illuminazione: fredda, blu-violacea, con lampi rossi intermittenti d'emergenza, altissimo
        contrasto. Stessa inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Alloggio/cabina privata di una stazione spaziale abbandonata, decadente in stile horror,
        illuminata da una singola striscia LED rossa d'emergenza che pulsa dal soffitto, sostituendo la
        lampadina nuda ma mantenendo la stessa dominante rossa intensa e inquietante.

        Un giaciglio/branda ripiegabile disfatta con coperte rosse spiegazzate, sulla cui superficie sono
        sparse le stesse bottiglie di vetro scuro vuote (o contenitori tecnologici simili), riviste e
        tablet rotti aperti. Il pavimento in grigliato metallico è coperto da altre bottiglie/contenitori
        rovesciati e schermi sparsi, suggerendo abbandono e disagio.

        A sinistra un oblò con schermo di protezione scostato, vetro scuro che non lascia intravedere
        nulla, e una consolle in metallo con un piccolo dispositivo acceso che lampeggia come un
        posacenere. A destra un armadietto metallico con un cassetto aperto e vuoto, sormontato da un
        contenitore pieno di mozziconi/detriti bruciati, e dietro uno scaffale con pannelli tecnologici
        e un cassetto leggermente aperto.

        Le pareti metalliche sono corrose, con crepe profonde e vistose che si diramano come circuiti
        danneggiati, pannelli scrostati e macchie di condensa.

        Illuminazione: dominante rossa intensa e pulsante dal LED d'emergenza, forti contrasti tra rosso
        vivido e ombre nere metalliche. Stessa inquadratura prospettica dal basso verso il giaciglio.
        """

        #cucina
        PROMPT4 = """
        Cucina/mensa di una stazione spaziale abbandonata, devastata, ambientazione molto buia, render
        3D fotorealistico in stile horror sci-fi. Composizione simmetrica centrata su un grande oblò a
        doppio pannello sul fondo, con vetri graffiati e sporchi da cui filtra una debole luce blu-grigia
        fredda dello spazio esterno, unica fonte di illuminazione, che crea un effetto di raggi
        volumetrici che penetrano nella stanza buia.

        Al centro, sotto l'oblò, un lavello modulare in metallo pieno di vassoi sporchi accatastati, con
        un panno giallo appeso al bordo. A destra del lavello, un rigeneratore/forno alimentare bianco
        con sportello, macchiato e corroso, con colature scure lungo la superficie. Sopra il lavello,
        pensili modulari scrostati con sportelli aperti, alcuni con contenitori di vetro scuro disposti
        sopra.

        Il pavimento in primo piano è interamente ricoperto di disordine: contenitori di vetro scuro
        vuoti sparsi, lattine schiacciate, detriti elettronici sparsi ovunque, pannelli rotti, e numerosi
        fogli/tablet sparpagliati. Al centro, una grande pozza di liquido che riflette debolmente la luce
        dell'oblò.

        Pareti metalliche scrostate e corrose ai lati, immerse quasi completamente nell'oscurità.

        Illuminazione: estremamente scura, dominata da ombre nere profonde, unica fonte blu-grigia fredda
        dallo spazio esterno. Stessa inquadratura frontale e simmetrica verso l'oblò.
        """

        #teatro (reinterpretato come sala di comando/auditorium della stazione)
        PROMPT5 = """
        Interno di una sala di comando/auditorium di una stazione spaziale abbandonata, vista da una
        balconata laterale verso il grande schermo centrale, render 3D fotorealistico, atmosfera
        tecnologica ma cupa e misteriosa. In primo piano, pavimento in pannelli metallici a griglia
        geometrica, illuminato da un fascio di luce soffusa blu che crea un punto luminoso centrale sul
        pavimento.

        Ai lati, due grandi colonne strutturali in metallo scuro, con basamenti tecnologici e condotti,
        sormontate da bracci robotici spenti e inattivi appesi come vecchi ganci. Le colonne incorniciano
        la scena in composizione simmetrica.

        Sullo sfondo centrale, un grande schermo curvo di comando, spento tranne per un bagliore rosso
        residuo, incorniciato da pannelli metallici ornati di cavi, racchiuso da una struttura scenica
        metallica. Davanti allo schermo, una piccola consolle rialzata.

        Tra il primo piano e il fondale, una balaustra metallica che separa la balconata dalla sala
        sottostante, dove si intravedono file di sedute di comando vuote. Ai lati, cabine laterali con
        pannelli di controllo spenti nell'ombra.

        Illuminazione: penombra fredda blu-violacea, illuminazione drammatica e soffusa, forte contrasto
        tra le zone in ombra e il punto luminoso centrale. Atmosfera solenne, tecnologica, inquietante.
        Stessa prospettiva centrale simmetrica.
        """

        #portone
        PROMPT6 = """
        Scena horror sci-fi con un grande portello blindato a due ante in primo piano, su sfondo
        completamente buio, render 3D fotorealistico. Il portello è realizzato in metallo corroso e
        ossidato, con grandi pannelli rettangolari rivettati e condotti tecnologici, texture di
        corrosione bruno-rossastra che si dirama in macchie irregolari, con colature verticali simili a
        sangue secco/ruggine lungo le ante.

        Al centro del portello, una grande sbarra/trave metallica orizzontale di blocco d'emergenza
        attraversa entrambe le ante, fissata con anelli metallici. Da questi anelli pendono cavi
        metallici che sostengono, al posto della bilancia tradizionale, due grandi moduli/contenitori
        tecnologici sospesi che pendono simmetricamente, leggermente sbilanciati, creando un elemento
        inquietante al centro.

        Sullo sfondo, in alto, un piccolo cono di luce bianca fredda da un faro d'emergenza illumina
        debolmente la parte superiore, con particelle di polvere e detriti sospesi in aria, mentre il
        resto rimane immerso nell'oscurità. Ai lati e in basso, cavi scoperti e fasci di fibre ottiche
        spente si estendono dagli angoli verso il centro.

        Il pavimento in primo piano è in grigliato metallico industriale. Le pareti laterali mostrano
        crepe profonde nei pannelli corrosi.

        Illuminazione: altissimo contrasto, buio quasi assoluto interrotto da luce bianco-fredda
        dall'alto. Atmosfera horror tecnologico, simbolica e inquietante. Stessa inquadratura frontale
        e perfettamente simmetrica.
        """

    elif Livel == 8:
        # deserto — palette arida, sabbia, ossa sbiancate dal sole, polvere
        #attrezzi
        PROMPT1 = """
        Officina/sala chirurgica improvvisata in una baracca del deserto abbandonata, stile horror,
        illuminata da una singola lampadina nuda che pende dal soffitto, la cui luce si mescola a fasci
        di luce polverosa che filtrano da crepe nelle pareti di legno e lamiera. Pareti in legno secco e
        lamiera arrugginita dal sole, sabbia accumulata negli angoli, crepe da cui filtra il vento caldo.

        Sulla parete di fondo, il pannello perforato mostra gli stessi utensili e lame arrugginiti, ora
        ricoperti da un sottile strato di polvere e sabbia dorata, che li rende opachi.

        Il banco di lavoro in legno graffiato e scolorito dal sole è coperto dalle stesse macchie di
        sangue secco, ora scurite e quasi fuse col legno arso, stesso tronchese al centro-sinistra,
        stesso straccio a destra, stesso bisturi in basso a destra, granelli di sabbia sparsi ovunque.

        Atmosfera: arida, calda, opprimente, palette ocra-beige-marrone desaturata, pulviscolo dorato
        sospeso nell'aria visibile nel fascio di luce, forte contrasto tra la luce calda e le ombre
        profonde. Fotorealistico, cinematografico, texture di legno secco, sabbia e metallo arrugginito
        ultra-dettagliate.

        Camera: identica inquadratura centrata e simmetrica verso il banco e la parete degli attrezzi.
        """

        #bagno
        PROMPT2 = """
        Bagno abbandonato in una stazione del deserto, ambientazione notturna arida, render 3D
        cinematografico. Stessa composizione centrata sulla toilette sporca, ora semi-sepolta da sabbia
        accumulata sul pavimento, water riempito di liquido rossastro scuro denso, con sabbia fine
        depositata sul bordo.

        Dietro, la stessa finestra a vetri multipli rotti, ora scheggiati e opacizzati dalla sabbia
        portata dal vento, rivela un cielo notturno desertico stellato con dune lontane invece di rami
        d'albero. Luce lunare fredda che filtra attraverso il pulviscolo sospeso come fonte principale.

        A sinistra lo specchio ovale incrinato, opacizzato da polvere e graffi di sabbia, sopra il
        lavabo vintage rigato di sangue e polvere. A destra la vasca vintage semi-sepolta da un cumulo
        di sabbia entrata dalla finestra, liquido denso e scuro che cola lungo il fianco impolverato.

        Pavimento con piastrelle rotte coperte da un velo di sabbia dorata, pozze di sangue secco
        mescolate a polvere, piccoli frammenti di ossa sbiancate dal sole sparsi visibili. Carta da
        parati vintage scolorita e scrostata dal caldo secco, macchiata di sangue.

        Illuminazione: cupa, calda-fredda contrastante, dominante ocra-blu desaturata. Stessa
        inquadratura architettonica grandangolare.
        """

        #camera da letto
        PROMPT3 = """
        Camera da letto decadente in una casa del deserto abbandonata, bagnata in una luce rossa intensa
        mescolata a toni polverosi ocra, render 3D fotorealistico. La lampadina nuda rossa pende dal
        soffitto, tingendo l'ambiente di rosso mentre la sabbia sospesa nell'aria crea un velo dorato-
        rossastro particolare.

        Letto disfatto con lenzuola rosse scolorite dal sole e coperte da un sottile strato di sabbia,
        sulla cui superficie sono sparse bottiglie di vetro scuro vuote, riviste e giornali ingialliti dal
        caldo. Il pavimento di legno secco e scricchiolante è coperto da altre bottiglie rovesciate,
        pagine sparse e sabbia accumulata negli angoli.

        A sinistra una finestra con tende pesanti consumate dal sole, vetri scuri graffiati dalla sabbia,
        e un comò in legno scolorito con un piccolo posacenere acceso. A destra un comodino con cassetto
        aperto e vuoto, posacenere pieno di mozziconi, scaffale in legno secco con un cassetto
        leggermente aperto.

        Le pareti sono scrostate e crepate dal caldo secco, con crepe profonde che si diramano come
        ragnatele, intonaco scrostato e polvere accumulata.

        Illuminazione: dominante rossa intensa mescolata a pulviscolo dorato sospeso, forti contrasti tra
        rosso vivido e ombre nere-marroni. Stessa inquadratura prospettica dal basso verso il letto.
        """

        #cucina
        PROMPT4 = """
        Cucina abbandonata in una casa del deserto, ambientazione notturna molto buia e arida, render 3D
        fotorealistico in stile horror/abbandono. Composizione simmetrica centrata su una finestra a
        doppio battente sul fondo, con vetri sporchi e graffiati dalla sabbia da cui filtra una debole
        luce blu-grigia fredda, unica fonte di illuminazione, con pulviscolo di sabbia visibile nel
        fascio volumetrico al posto della semplice polvere.

        Al centro, sotto la finestra, un lavello da cucina vintage in ceramica bianca ingiallita dal
        sole, pieno di piatti sporchi accatastati, con un canovaccio giallo scolorito appeso al bordo.
        A destra, una vecchia cucina a gas bianca arrugginita dal caldo secco, con colature scure lungo
        la superficie. Pensili scrostati e sbiaditi dal sole, con sportelli aperti.

        Il pavimento in primo piano è ricoperto di disordine — bottiglie, lattine schiacciate, mozziconi,
        vetri rotti, giornali ingialliti — con un sottile strato di sabbia depositata ovunque. Al centro,
        una grande pozza di liquido che riflette debolmente la luce della finestra, sopra pagine
        consumate dal caldo.

        Pareti scrostate e sbiadite dal sole ai lati, immerse quasi completamente nell'oscurità.

        Illuminazione: scura e cupa, dominata da ombre nere profonde, unica fonte blu-grigia fredda dalla
        finestra. Stessa inquadratura frontale e simmetrica.
        """

        #teatro
        PROMPT5 = """
        Antico teatro coloniale nel deserto, ormai abbandonato e invaso dalla sabbia, vista da un
        palco/balconata laterale verso il sipario centrale, render 3D fotorealistico, atmosfera elegante
        ma cupa e polverosa. In primo piano, pavimento in parquet di legno a intarsio geometrico, ora
        scolorito e coperto da un velo di sabbia fine, illuminato da un fascio di luce soffusa dorata
        che crea un punto luminoso centrale.

        Ai lati, due grandi colonne in legno massiccio scuro, scolorite e crepate dal caldo secco, con
        basamenti scolpiti coperti di polvere, sormontate da mensole in metallo nero ossidato con
        gancetti in ferro battuto arrugginito.

        Sullo sfondo centrale, il grande sipario di velluto rosso bordeaux appare sbiadito e consumato
        dal sole, con drappeggi impolverati bordati di frange dorate ormai opache, racchiuso da una
        cornice scenica dorata scrostata.

        Balaustra in legno tornito coperta di sabbia, platea sottostante con poltrone rosse scolorite,
        palchi laterali semi-sepolti da dune di sabbia entrate da qualche apertura.

        Illuminazione: penombra calda con toni ocra e dorati polverosi, illuminazione drammatica e
        soffusa, forte contrasto tra ombre e il punto luminoso centrale. Stessa prospettiva centrale
        simmetrica verso il sipario.
        """

        #portone
        PROMPT6 = """
        Scena horror/dark fantasy nel deserto con un grande portone metallico a due ante in primo piano,
        su sfondo completamente buio, render 3D fotorealistico. Il portone è in metallo arrugginito e
        ossidato dal sole e dalla sabbia, con grandi pannelli rettangolari rivettati, texture di ruggine
        color bruno-rossastro mista a graffi di sabbia, con colature verticali simili a sangue secco che
        scendono lungo le ante, cumuli di sabbia accumulati alla base.

        Al centro del portone, la stessa grande sbarra/trave metallica orizzontale di blocco, fissata con
        anelli metallici corrosi. Da questi anelli pendono catene arrugginite che sostengono la stessa
        bilancia a doppio piatto, con i due piatti metallici parzialmente riempiti di sabbia, leggermente
        sbilanciati.

        Sullo sfondo, in alto, un piccolo cono di luce bianca fredda e polverosa illumina debolmente la
        parte superiore, con particelle di sabbia sospese in aria al posto della semplice polvere, mentre
        il resto rimane immerso nell'oscurità. Ai lati, ragnatele fitte coperte di sabbia si estendono
        dagli angoli verso il centro.

        Il pavimento in primo piano è in pietra/cemento grezzo semi-sepolto da dune di sabbia. Le pareti
        laterali mostrano crepe profonde nell'intonaco scuro screpolato dal caldo.

        Illuminazione: altissimo contrasto, buio quasi assoluto interrotto da una sottile luce dall'alto,
        toni freddi in alto contro i toni caldi ruggine-sabbia del portone. Stessa inquadratura frontale
        e perfettamente simmetrica.
        """


     

# Mappa ordinata: numero -> (nome_base, nome_file_input, prompt)
LOCATIONS = {
    1: ("attrezzi", "attrezzi.png", PROMPT1),
    2: ("bagno", "bagno.png", PROMPT2),
    3: ("camera da letto", "camera da letto.png", PROMPT3),
    4: ("cucina", "cucina.png", PROMPT4),
    5: ("parche", "parche.png", PROMPT5),
    6: ("portone", "portone.png", PROMPT6),
}


def genera_location_iniziali():
    global LOCATIONS, PROMPT1, PROMPT2, PROMPT3, PROMPT4, PROMPT5, PROMPT6

    prompts()  # aggiorna PROMPT1..PROMPT6 in base al Livel corrente

    # fix: ricostruisci il dizionario DOPO aver aggiornato i prompt,
    # altrimenti contiene sempre le stringhe vuote iniziali
    LOCATIONS = {
        1: ("attrezzi", "attrezzi.png", PROMPT1),
        2: ("bagno", "bagno.png", PROMPT2),
        3: ("camera da letto", "camera da letto.png", PROMPT3),
        4: ("cucina", "cucina.png", PROMPT4),
        5: ("parche", "parche.png", PROMPT5),
        6: ("portone", "portone.png", PROMPT6),
    }

    """Genera tutte le location necessarie all'avvio, se non esistono già."""
    for k in tqdm(range(1, 7), desc="generazioni locations"):
        base_name, input_filename, prompt = LOCATIONS[k]

        out_path = os.path.join(out_dir, f"{base_name}_flux.png")
        p_image1 = os.path.join(out_dir, input_filename)

        if not os.path.exists(out_path):
            flux2(
                prompt,
                DEFAULT_STEPS,
                p_image1,
                path_image2,
                path_image3,
                path_image4,
                path_lora,
                canvas_w,
                canvas_h,
                out_dir,
                name=None
            )


# --- Genera SUBITO all'avvio, prima di mostrare la finestra interattiva ---
genera_location_iniziali()

# --- Costruiamo la matrice delle stanze ---
matrix_stanze = [
    [None,                       "portone_flux.png",        None],
    ["cucina_flux.png",          "parche_flux.png",          "attrezzi_flux.png"],
    ["camera da letto_flux.png", "bagno_flux.png",           None],
]

# stanza iniziale: cucina -> riga 1, colonna 0
stanza_attuale = matrix_stanze[1][0]

# ============================================================
# LAYOUT FINESTRA PRINCIPALE
# ============================================================

window = TkinterDnD.Tk()
window.title("Severed Blood")
window.geometry("1680x900")
window.resizable(False, False)
window.config(background='gray')

# Calcola la posizione per centrare la finestra in alto
window.update_idletasks()  # assicura che le dimensioni siano aggiornate
w = 1750
h = 940
screen_w = window.winfo_screenwidth()
screen_h = window.winfo_screenheight()
x = int((screen_w - w) / 2)
y = 0  # in alto
window.geometry(f"{w}x{h}+{x}+{y}")

window.lift()
window.columnconfigure(0, weight=0)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=0)
window.rowconfigure(0, weight=1)

# ============================================================
# COLONNA 0 — Bottoni New/Load/Save
# ============================================================

frame_button = tk.Frame(window, bg='gray')
frame_button.grid(row=0, column=0, sticky='n', padx=(10, 0), pady=10)


import os
import shutil
import subprocess
from tkinter import filedialog, messagebox

WINRAR_EXE = r"C:\Program Files\WinRAR\WinRAR.exe"  # verifica il percorso reale sul tuo sistema
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cartella dello script, indipendente da dove viene lanciato


def f_new_game():
    conferma = messagebox.askyesno(
        "Nuova Partita",
        "Iniziare una nuova partita cancellerà i progressi attuali non salvati "
        "(personaggi, storia, location). Continuare?"
    )
    if not conferma:
        return

    for f in ("indice.txt", "use_character.txt", f"avvia_gioco.txt","livel.txt"):
        if os.path.exists(f):
            os.remove(f)

    for d in ("character", "history", "location"):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)  # ricrea le cartelle vuote, pronte per la nuova partita

    messagebox.showinfo("Nuova Partita", "Partita azzerata. Pronta per iniziare.")


def f_loadgame():
    path_s = 'salvataggi'
    os.makedirs(path_s, exist_ok=True)

    if not os.path.exists(WINRAR_EXE):
        messagebox.showerror("Errore", "WinRAR non trovato sul sistema. Impossibile caricare il salvataggio.")
        return

    path_name = filedialog.askopenfilename(
        initialdir=path_s,
        defaultextension=".rar",
        filetypes=[("Archivio RAR", "*.rar")]
    )

    if not path_name:
        return  # l'utente ha annullato la finestra di dialogo

    if not os.path.exists(path_name):
        messagebox.showerror("Errore", "Il file di salvataggio selezionato non esiste.")
        return

    sovrascrivi = messagebox.askyesno(
        "Attenzione",
        "Caricare questo salvataggio sovrascriverà i file di gioco attuali "
        "(personaggi, storia, location). Continuare?"
    )
    if not sovrascrivi:
        return

    # x    = estrai mantenendo la struttura delle cartelle dentro l'archivio
    # -o+  = sovrascrivi sempre i file esistenti senza chiedere conferma
    # -y   = rispondi automaticamente "sì" a qualsiasi altra richiesta di WinRAR
    comando = [WINRAR_EXE, "x", "-o+", "-y", path_name, BASE_DIR + os.sep]
    risultato = subprocess.run(comando, capture_output=True, text=True)

    if risultato.returncode != 0:
        messagebox.showerror("Errore", f"Caricamento fallito:\n{risultato.stderr}")
    else:
        messagebox.showinfo("Completato", "Salvataggio caricato correttamente.")


def f_savegame():
    path_s = 'salvataggi'
    os.makedirs(path_s, exist_ok=True)

    if not os.path.exists(WINRAR_EXE):
        messagebox.showerror("Errore", "WinRAR non trovato sul sistema. Impossibile creare il salvataggio in formato .rar.")
        return

    path_name = filedialog.asksaveasfilename(
        initialdir=path_s,
        defaultextension=".rar",
        filetypes=[("Archivio RAR", "*.rar")]
    )

    if not path_name:
        return

    if os.path.exists(path_name):
        sovrascrivi = messagebox.askyesno(
            "Attenzione",
            "Il file di salvataggio esiste già. Sovrascriverlo?"
        )
        if not sovrascrivi:
            return
        os.remove(path_name)  # WinRAR in modalità "a" aggiunge, non sovrascrive: va rimosso prima

    elementi_da_comprimere = [
        f for f in ("indice.txt", "use_character.txt", "avvia_gioco.txt","livel.txt")
        if os.path.exists(f)
    ] + [
        d for d in ("character", "history", "location")
        if os.path.isdir(d)
    ]

    if not elementi_da_comprimere:
        messagebox.showwarning("Attenzione", "Non c'è nessuna partita in corso da salvare.")
        return

    comando = [WINRAR_EXE, "a", "-r", path_name] + elementi_da_comprimere
    risultato = subprocess.run(comando, capture_output=True, text=True)

    if risultato.returncode != 0:
        messagebox.showerror("Errore", f"Compressione fallita:\n{risultato.stderr}")
    else:
        messagebox.showinfo("Completato", "Partita salvata correttamente.")


new_game = tk.Button(frame_button, text='New Game', width=14, height=2, bg='lightgray', command=f_new_game)
new_game.grid(row=0, column=0, pady=5, padx=5)

load = tk.Button(frame_button, text='Load Game', width=14, height=2, bg='lightgray', command=f_loadgame)
load.grid(row=1, column=0, pady=5, padx=5)

save = tk.Button(frame_button, text='Save Game', width=14, height=2, bg='lightgray', command=f_savegame)
save.grid(row=2, column=0, pady=5, padx=5)


def che_ver():
    if face_var.get():
        full_var.set(False)
    elif full_var.get():
        face_var.set(False)


face_var = tk.BooleanVar()
full_var = tk.BooleanVar()

face_check = tk.Checkbutton(frame_button, text='crop_face', variable=face_var, command=che_ver)
face_check.grid(row=3, column=0, pady=5)

full_check = tk.Checkbutton(frame_button, text='full', variable=full_var, command=che_ver)
full_check.grid(row=4, column=0, pady=5)

import tkinter as tk

hystor_indice=tk.Label(frame_button, text='Hystor_Indice').grid(row=5, column=0, pady=5)
# ✅ Aggiungi questa funzione PRIMA dello Spinbox
def on_indice_change(*args):
    """Callback quando cambia l'indice nello spinbox"""
    global stanza_attuale, tk_img_attuale, img_attuale, frame, lab_stanza, canvas_w, canvas_h
    
    try:
        print(f"📊 Indice cambiato: {var_indice.get()}")
        
        # ✅ Carica l'immagine con il nuovo indice
        path_stanza = get_stanza_path(stanza_attuale)
        
        if os.path.exists(path_stanza):
            img_attuale = Image.open(path_stanza).convert("RGB")
            img_attuale = img_attuale.resize((canvas_w, canvas_h))
            tk_img_attuale = ImageTk.PhotoImage(img_attuale)
            
            frame.delete("all")
            frame.create_image(0, 0, anchor='nw', image=tk_img_attuale)
            lab_stanza.config(text=f"Stanza: {os.path.basename(path_stanza)}")
            lab_stanza.update_idletasks()
            print(f"✅ Canvas aggiornata: {path_stanza}")
        else:
            print(f"❌ Immagine non trovata: {path_stanza}")
    
    except Exception as e:
        print(f"❌ Errore cambio indice: {e}")
        import traceback
        traceback.print_exc()

# ✅ Aggiungi PRIMA dello Spinbox
var_indice = tk.IntVar(value=1)

spinbox_indice = tk.Spinbox(
    frame_button,
    from_=1,        # valore minimo
    to=10000,         # valore massimo
    increment=1,    # incremento ad ogni click freccia
    width=5,
    font=("Arial", 10),
    justify="center",
    textvariable=var_indice
)
spinbox_indice.grid(row=6, column=0, pady=5)

# ✅ COLLEGA il callback quando var_indice cambia
var_indice.trace('w', on_indice_change)


import threading as t
import os
from PIL import Image


def dividi_sfondo(sfondo_path, out_dir, n_parti=4):
    """
    Divide l'immagine di sfondo in n_parti strisce verticali uguali
    (larghezza = sfondo.width // n_parti), salvandole come file separati.
    L'ultima striscia assorbe eventuali pixel avanzati dalla divisione intera,
    cosi' la somma delle larghezze torna sempre esatta.
    Ritorna la lista dei path delle strisce create, in ordine sinistra->destra.
    """
    img = Image.open(sfondo_path).convert("RGB")
    w, h = img.size
    parte_w = w // n_parti

    paths = []
    for i in range(n_parti):
        x0 = i * parte_w
        x1 = (i + 1) * parte_w if i < n_parti - 1 else w  # l'ultima prende il resto
        striscia = img.crop((x0, 0, x1, h))

        out_path = os.path.join(out_dir, f"sfondo_parte{i+1}.png")
        striscia.save(out_path)
        paths.append(out_path)
        print(f"striscia {i+1}/{n_parti} salvata: {out_path} ({striscia.width}x{striscia.height})")

    return paths


def ricostruisci_collage(paths_generati, out_path):
    """
    Riaffianca orizzontalmente le n immagini generate (una per ogni striscia)
    per ricreare l'immagine finale completa. Presuppone che ogni immagine
    generata abbia la stessa altezza (quella dello sfondo originale).
    """
    immagini = [Image.open(p).convert("RGB") for p in paths_generati]

    h = immagini[0].height
    w_totale = sum(im.width for im in immagini)

    finale = Image.new("RGB", (w_totale, h), (0, 0, 0))
    x_offset = 0
    for im in immagini:
        if im.height != h:
            # sicurezza: se un pezzo ha altezza diversa, lo scala per allinearlo
            ratio = h / im.height
            im = im.resize((int(im.width * ratio), h), Image.LANCZOS)
        finale.paste(im, (x_offset, 0))
        x_offset += im.width

    finale.save(out_path)
    print(f"immagine finale ricomposta salvata in: {out_path}")
    return out_path





def f_avvia_gioco():
    print("avvia gioco")

    out_dir = "history"
    os.makedirs(out_dir, exist_ok=True)

    personaggio1 = r"character\ch1full.png"
    personaggio2 = r"character\ch2full.png"
    personaggio3 = r"character\ch3full.png"
    personaggio4 = r"character\ch4full.png"
    personaggi = [personaggio1, personaggio2, personaggio3, personaggio4]

    scena_sfondo = os.path.join("location", "cucina_flux.png")

    out_path_finale = os.path.join(out_dir, "cucina_flux1.png")  # nome temporaneo di lavoro

    strisce_paths = [os.path.join(out_dir, f"sfondo_parte{i+1}.png") for i in range(4)]
    parti_generate_paths = [os.path.join(out_dir, f"parte{i+1}_generata.png") for i in range(4)]


    if not os.path.exists(scena_sfondo):
        print(f"ATTENZIONE: sfondo non trovato: {scena_sfondo}")
        return

    # --- 1. dividi lo sfondo in 4 strisce verticali (se non già fatto) ---
    if all(os.path.exists(p) for p in strisce_paths):
        print("strisce di sfondo già esistenti, riuso quelle salvate")
    else:
        print("divisione dello sfondo in 4 parti...")
        strisce_paths = dividi_sfondo(scena_sfondo, out_dir, n_parti=4)

    # --- 2. per ogni striscia genera la ragazza corrispondente ---
    prompt_base = """Aggiungi la ragazza dell'image 2 DISTESA IN PROFONDITÀ sul pavimento visibile nell'image 1:
- LA TESTA DELLA RAGAZZA DEVE ESSERE IN PRIMO PIANO, RIVOLTA VERSO LO SPETTATORE
- I PIEDI DEVONO ESSERE ALLONTANATI VERSO IL FONDO DELLA STANZA (in profondità, non lateralmente)
- IL CORPO È COMPLETAMENTE SDRAIATO DA DAVANTI A DIETRO, NON DA SINISTRA A DESTRA
- La ragazza occupa lo spazio in profondità della scena, come in una prospettiva che va dal primo piano al fondo
- TESTA E PIEDI COMPLETAMENTE VISIBILI: non tagliare mai testa, capelli, gambe, piedi o nessuna parte del corpo
- La ragazza NON deve essere orizzontale sinistra-destra, ma verticale davanti-dietro in profondità

Mantieni la massima coerenza con l'immagine di riferimento: stesso viso, stessi occhi, stessi capelli (colore, lunghezza, taglio), stessa corporatura e stesso outfit esatto mostrato nell'image 2.
Mantieni invariato tutto il resto della scena: illuminazione, prospettiva, arredamento, oggetti sul pavimento e atmosfera. 
Non modificare nulla dello sfondo oltre ad aggiungere la ragazza.

DA EVITARE ASSOLUTAMENTE: numeri, testo, watermark, bordi; arti tagliate, dita sbagliate, arti in più, anatomia deformata, proporzioni sbagliate, volti distorti, mani deformi. Il corpo deve essere COMPLETO E INTEGRO."""
    for i, (striscia_path, personaggio) in enumerate(zip(strisce_paths, personaggi)):
        name = f"parte{i+1}_generata"
        out_parte_path = parti_generate_paths[i]

        if os.path.exists(out_parte_path):
            print(f"salto step {i+1}/4, esiste già: {out_parte_path}")
            continue

        print(f"generazione ragazza {i+1}/4 sulla striscia {striscia_path}")

        with Image.open(striscia_path) as im_striscia:
            wc_parte, hc_parte = im_striscia.size

        try:
            flux2(
                prompt_base,
                steps=DEFAULT_STEPS,
                path_image1=striscia_path,
                path_image2=personaggio,
                path_image3=None,
                path_image4=None,
                path_lora=None,
                wc=wc_parte,
                hc=hc_parte,
                out_dir=out_dir,
                name=name
            )
        except Exception as e:
            print(f"Errore durante la generazione della parte {i+1}: {e}")

    # --- 3. ricostruisci il collage finale affiancando le 4 parti generate ---
    if all(os.path.exists(p) for p in parti_generate_paths):
        print("ricostruzione immagine finale dalle 4 parti generate...")
        ricostruisci_collage(parti_generate_paths, out_path_finale)

    else:
        mancanti = [p for p in parti_generate_paths if not os.path.exists(p)]
        print(f"ATTENZIONE: mancano alcune parti generate, impossibile ricomporre: {mancanti}")

    print("gioco avviato: generazione completata")

import os

AVVIA_GIOCO_FILE = "avvia_gioco.txt"

# ------------------------------------------------------------
# RICARICA VALORE SALVATO IN FILE (all'avvio dell'app)
# ------------------------------------------------------------
def carica_avvia_gioco():
    if os.path.exists(AVVIA_GIOCO_FILE):
        with open(AVVIA_GIOCO_FILE) as f:
            valore = f.readline().strip()
        return valore == "1"
    return False

def salva_avvia_gioco(valore: bool):
    with open(AVVIA_GIOCO_FILE, "w") as f:
        f.write("1" if valore else "0")

AVVIA_GIOCO = carica_avvia_gioco()


def avvia_gioco_thread():
    global AVVIA_GIOCO
    avvia_gioco.config(state="disabled")  # evita doppio click mentre genera

    def run():
        global AVVIA_GIOCO
        try:
            AVVIA_GIOCO = True
            salva_avvia_gioco(AVVIA_GIOCO)   # salva subito, prima di generare
            f_avvia_gioco()
        finally:
            avvia_gioco.config(state="normal")

    t.Thread(target=run, daemon=True).start()


avvia_gioco = tk.Button(
    frame_button,
    text='Avvia Gioco',
    width=14,
    height=2,
    bg='lightgray',
    command=avvia_gioco_thread
)
avvia_gioco.grid(row=7, column=0, pady=5)

# opzionale: se il gioco era già stato avviato in precedenza,
# puoi disabilitare il bottone o cambiarne il testo all'avvio dell'app
if AVVIA_GIOCO:
    avvia_gioco.config(text="Riprendi Gioco")

# Variabili di selezione (mutuamente esclusive)
select_face_var = tk.BooleanVar(value=False)
select_body_var = tk.BooleanVar(value=True)

def on_face_select():
    if select_face_var.get():
        select_body_var.set(False)

def on_body_select():
    if select_body_var.get():
        select_face_var.set(False)

frame_chek = tk.Frame(frame_button)
frame_chek.grid(row=8, column=0, sticky='w')

select_face = tk.Checkbutton(frame_chek, text="Face", variable=select_face_var, 
                             command=on_face_select)
select_face.grid(row=0, column=0, sticky='w')

select_body = tk.Checkbutton(frame_chek, text="Body", variable=select_body_var, 
                             command=on_body_select)
select_body.grid(row=0, column=1, sticky='w')






# ============================================================
# COLONNA 1 — Canvas + Testo insieme nello stesso frame
# ============================================================

# Funzione per ottenere il percorso corretto della stanza
def get_stanza_path(nome_stanza):
    """Controlla se esiste in history, altrimenti usa location"""
    indice = spinbox_indice.get()
    
    # Estrai il nome senza estensione
    nome_base = nome_stanza.split('.')[0]
    
    # Prova prima in history con l'indice
    history_path = f"./history/{nome_base}{indice}.png"
    if os.path.exists(history_path):
        return history_path
    
    # Altrimenti usa location
    location_path = f"./location/{nome_stanza}"
    return location_path

frame_centro = tk.Frame(window, bg='gray')
frame_centro.grid(row=0, column=1, sticky='n', pady=10)

frame = tk.Canvas(frame_centro, width=canvas_w, height=canvas_h, bg='black')
frame.grid(row=0, column=0)


oggetto_selezionato = None

def RESIZE_O(img):
    dim = 128
    wo, ho = img.size

    if wo >= ho:
        ho = (dim * ho) // wo
        wo = dim
    else:
        wo = (dim * wo) // ho
        ho = dim

    img = img.resize((wo, ho), Image.BICUBIC)
    return img

# Percorsi degli oggetti (unica fonte di verita': l'ordine qui determina
# l'ordine sia delle immagini che dei path, cosi' restano sempre allineati)
oggetti_path = [
    "oggetti/cacciavite.png",
    "oggetti/martello.png",
    "oggetti/sedia.png",
    "oggetti/seghetto.png",
]

# Carica e ridimensiona gli oggetti a partire dagli stessi path
oggetti = [RESIZE_O(Image.open(p).convert("RGBA")) for p in oggetti_path]

# Converti in PhotoImage
oggetti_photo = []
for obj in oggetti:
    oggetti_photo.append(ImageTk.PhotoImage(obj))

ind = 0
oggetto_id = None
mouse_pos = (0, 0)

def cambia_oggetto(event):
    global lab_oggetto
    """Cambia oggetto quando scorri (MouseWheel)"""
    global ind, oggetto, frame, oggetto_id, mouse_pos, oggetto_selezionato, testo

    if not oggetto:
        return

    mouse_pos = (event.x, event.y)

    if oggetto_id is not None:
        frame.delete(oggetto_id)

    # Salva PRIMA di incrementare
    oggetto_selezionato = oggetti_path[ind]
    print(f"Oggetto selezionato: {oggetto_selezionato}")

    # Disegna l'oggetto
    oggetto_id = frame.create_image(
        mouse_pos[0], mouse_pos[1],
        image=oggetti_photo[ind],
        anchor='sw'
    )

    # Incrementa DOPO
    ind = (ind + 1) % len(oggetti)
    frame.update_idletasks()

    # nome pulito senza estensione, usato sia per la label che per il confronto
    nome_oggetto = os.path.basename(oggetto_selezionato).split('.')[0]

    lab_oggetto.config(text=f"Oggetto: {nome_oggetto}")
    lab_oggetto.update_idletasks()

    # qui sceglie solo il prompt in base all'oggetto
    testo.delete('1.0', tk.END)
    p = ''

    if nome_oggetto == 'sedia':
        p = """Posiziona il trono del image 2 gotico nero con dettagli di teschio sul parquet davanti al balcone teatrale del image 1, stile horror gotico, atmosfera lugubre e sinestra"""

    # utente decide quale prompt usare
    elif nome_oggetto == 'cacciavite':
        p = """prompt1:aggiungi la (ferita del image 2 sulla gamba destra della ragazza del image 1) distesa sul tavolo degli attrezzi.
aggiungi (il cacciavite nero e giallo sporco di sangue del image 3, poggiato sul tavolo degli attrezzi del image 1,
Mantieni massima coerenza della ragazza del image 1: viso, occhi, capelli,capelli colore nero, del fisico e del outfit stesso vestito.
Mantieni massima coerenza con il cacciavite nero e giallo del image 3, stessa forma e stessi colori del manico.

prompt2:aggiungi la (ferita con gli occhi mancanti del image 2 sulla faccia della ragazza del image 1) distesa sul tavolo degli attrezzi.
viso privo di occhi, orbite oculari visibili e profonde.
aggiungi (il cacciavite nero e giallo sporco di sangue del image 3, poggiato sul tavolo degli attrezzi del image 1);
Mantieni massima coerenza della ragazza del image 1: viso,capelli,capelli colore nero, del fisico e del outfit stesso vestito.
Mantieni massima coerenza con il cacciavite nero e giallo del image 3, stessa forma e stessi colori del manico"""

    elif nome_oggetto == 'martello':
        p = """aggiungi:  (la ferita del image 2 sulla fronte) della ragazza, distesa sul tavolo nel image 1;
aggiungi: (la mazzetta blu e nera del image 3 sporca di sangue) poggiata sul tavolo degli attrezzi del image 1
Mantieni massima coerenza della ragazza del image 1: viso, occhi, capelli ,capelli colore nero, del fisico e del outfit stesso vestito.;
Mantieni massima coerenza della mazzetta nel image 3"""

    elif nome_oggetto == 'seghetto':
        p = """aggiungi:  (la ferita del image 2 sulla gamba destra) della ragazza, distesa sul tavolo nel image 1;
aggiungi: (il seghetto azzurro del image 3 sporco di sangue) poggiato sul tavolo degli attrezzi del image 1
Mantieni massima coerenza della ragazza del image 1: viso, occhi, capelli,capelli colore nero, del fisico e del outfit stesso vestito.;
Mantieni massima coerenza del seghetto nel image 3"""

    

    testo.insert('1.0', p)


# PROVA ENTRAMBI GLI EVENTI
frame.bind("<MouseWheel>", cambia_oggetto)


oggetto = False

def f_oggetto():
    global oggetto, oggetto_id, oggetto_selezionato, path_stanza

    oggetto = not oggetto   # fix: era una tupla, ora è un booleano vero e proprio

    if oggetto:
        buttonobject.config(bg='light green', text='🎯 Oggetto')
        print("✓ Modalità oggetto ATTIVA")
        frame.focus_set()
    else:
        if oggetto_id is not None:
            frame.delete(oggetto_id)
            oggetto_id = None
            oggetto_selezionato = None

        path_stanza = get_stanza_path(path_stanza)  # ora funziona: path_stanza è global
        p = ''

        nome_file = os.path.basename(path_stanza)  # fix: rimosso [0], serve l'intero nome file

        if 'attrezzi' in nome_file:
            p = """la ragazza con IL VESTITO ROSSO del image 2 è distessa in orizzontale sul tavolo degli attrezzi del image 1, nel inquadratura del tavolo sono entrambi visibili i piedi nudi e la testa; mantieni proporzioni del corpo della ragazza;
le braccia sono entrambi distesse lungo il corpo della ragazza verso i piedi, le gambe sono entrambi distesse sul tavolo. occhi aperti, sguardo assente.
Mantieni massima coerenza della ragazza nel image 2 ,del suo viso, dei suoi occhi, della sua capigliatura, del suo fisico, e del outfit."""
        elif 'parche' in nome_file:   # fix: elif invece di if
            p = """aggiungi la ragazza con il vestito rosso del image 2 seduta sul trono di rovere del image 1 , mantieni massima coerenza con il personaggio del image 1 del viso , 
dei capelli ,occhi fisico e outfit"""

        testo.delete('1.0', 'end')
        testo.insert('1.0', p)
        buttonobject.config(bg='saddlebrown', text='○ Oggetto')
        print("✗ Modalità oggetto DISATTIVA")

buttonobject = tk.Button(
    frame_button, 
    text='○ Oggetto', 
    bg='saddlebrown',
    fg='white',
    font=('Arial', 10, 'bold'),
    command=f_oggetto
)
buttonobject.grid(row=9, column=0, pady=3)


stanza_attuale = matrix_stanze[1][0]
img_attuale = Image.open(get_stanza_path(stanza_attuale)).convert("RGB")
img_attuale = img_attuale.resize((canvas_w, canvas_h))
tk_img_attuale = ImageTk.PhotoImage(img_attuale)
frame.create_image(0, 0, anchor='nw', image=tk_img_attuale)
frame.update_idletasks()

# Mappa dei collegamenti
collegamenti = {
    matrix_stanze[1][0]: {"dx": matrix_stanze[1][1]},
    matrix_stanze[1][1]: {
        "dx": matrix_stanze[1][2],
        "sx": matrix_stanze[1][0],
        "su": matrix_stanze[0][1],
        "giu": matrix_stanze[2][1],
    },
    matrix_stanze[1][2]: {"sx": matrix_stanze[1][1]},
    matrix_stanze[2][0]: {"dx": matrix_stanze[2][1]},
    matrix_stanze[2][1]: {
        "sx": matrix_stanze[2][0],
        "su": matrix_stanze[1][1],
    },
    matrix_stanze[0][1]: {"giu": matrix_stanze[1][1]},
}
path_stanza = None
import time
import math
import os
from PIL import Image, ImageTk, ImageOps, ImageDraw  # ✅ Aggiungi ImageDraw

# ============================================================
# FUNZIONE TINTA CORDA (definita PRIMA di essere usata)
# ============================================================
def tinteggia_corda(img_rgba, colore):
    """Applica una tinta colorata mantenendo la texture/luminosità originale della corda.
    colore = tupla RGB, es. (80, 170, 255) per azzurro"""
    r, g, b, a = img_rgba.split()
    grigio = Image.merge("RGB", (r, g, b)).convert("L")
    tinta = ImageOps.colorize(grigio, black=(0, 0, 0), white=colore)
    tinta.putalpha(a)
    return tinta

# ============================================================
# LAZO DEL SEGHETTO (simile al lazo della corda)
# ============================================================

# Crea rope image SCURA per il seghetto
rope_img_seghetto = Image.open("lazo.png").convert("RGBA")
scala_seghetto = 0.35
nuova_w_seghetto = int(rope_img_seghetto.width * scala_seghetto)
nuova_h_seghetto = int(rope_img_seghetto.height * scala_seghetto)
rope_img_seghetto = rope_img_seghetto.resize((nuova_w_seghetto, nuova_h_seghetto), Image.LANCZOS)

# Colora il seghetto di BLU SCURO
rope_img_seghetto = tinteggia_corda(rope_img_seghetto, (0, 100, 200))

# Pulisci gli alpha
pixel_data_seghetto = rope_img_seghetto.get_flattened_data()
nuovi_pixel_seghetto = [(r, g, b, 0) if r > 235 and g > 235 and b > 235 else (r, g, b, 255)
                        for r, g, b, a in pixel_data_seghetto]
rope_img_seghetto.putdata(nuovi_pixel_seghetto)

rope_w_seghetto, rope_h_seghetto = rope_img_seghetto.size
rope_cache_seghetto = {}

def get_rope_tile_seghetto(angolo_gradi):
    """Ottiene un tile di seghetto ruotato"""
    angolo_arrotondato = round(angolo_gradi / 5) * 5
    if angolo_arrotondato not in rope_cache_seghetto:
        ruotata = rope_img_seghetto.rotate(-angolo_arrotondato, expand=True, resample=Image.BICUBIC)
        rope_cache_seghetto[angolo_arrotondato] = ImageTk.PhotoImage(ruotata)
    return rope_cache_seghetto[angolo_arrotondato]

# STATO DEL SEGHETTO
punti_seghetto = []
tile_info_seghetto = []
ultimo_tile_pos_seghetto = None
tile_refs_seghetto = []


def seghetto_attivo_ora():
    """Vero quando l'oggetto selezionato è il seghetto e non siamo in cucina.
    Sostituisce il vecchio flag lazo_seghetto_attivo (mai attivato)."""
    return (oggetto_selezionato is not None
            and 'seghetto' in os.path.basename(oggetto_selezionato).lower()
            and 'cucina' not in os.path.basename(path_stanza).lower())


def f_lazo_seghetto_start(event):
    """Inizia il tracciamento del seghetto"""
    global punti_seghetto, ultimo_tile_pos_seghetto, tile_refs_seghetto, tile_info_seghetto
    global oggetto_id, mouse_pos
    if not seghetto_attivo_ora():
        return
    frame.delete("seghetto_lazo")
    tile_refs_seghetto = []
    tile_info_seghetto = []
    punti_seghetto = [(event.x, event.y)]
    ultimo_tile_pos_seghetto = (event.x, event.y)

    # ✅ Porta subito l'icona del seghetto sotto il cursore
    mouse_pos = (event.x, event.y)
    if oggetto_id is not None:
        frame.coords(oggetto_id, event.x, event.y)
        frame.tag_raise(oggetto_id)

    print("🔪 Seghetto: tracciamento iniziato")


def f_draw_lazo_seghetto(event):
    """Disegna il lazo del seghetto durante il movimento e fa 'compositing':
    l'icona del seghetto resta sempre sopra ai tile del lazo, seguendo il cursore."""
    global ultimo_tile_pos_seghetto, punti_seghetto, tile_refs_seghetto, tile_info_seghetto
    global oggetto_id, mouse_pos

    if not seghetto_attivo_ora() or not punti_seghetto:
        return

    x0, y0 = ultimo_tile_pos_seghetto
    dx, dy = event.x - x0, event.y - y0
    distanza = math.hypot(dx, dy)
    spaziatura = rope_w_seghetto * 0.75

    if distanza >= spaziatura:
        angolo = math.degrees(math.atan2(dy, dx))
        cx, cy = x0 + dx / 2, y0 + dy / 2

        tile_info_seghetto.append((cx, cy, angolo))
        tile = get_rope_tile_seghetto(angolo)
        frame.create_image(cx, cy, image=tile, tags="seghetto_lazo")
        tile_refs_seghetto.append(tile)

        ultimo_tile_pos_seghetto = (event.x, event.y)
        punti_seghetto.append((event.x, event.y))

        print(f"🔪 Seghetto: {len(punti_seghetto)} punti")

    # ✅ L'icona del seghetto segue SEMPRE il cursore e resta sopra i tile appena creati
    mouse_pos = (event.x, event.y)
    if oggetto_id is not None:
        frame.coords(oggetto_id, event.x, event.y)
        frame.tag_raise(oggetto_id)


def ridisegna_lazo_seghetto():
    """Ridisegna il lazo del seghetto (es. quando cambia colore)"""
    global tile_refs_seghetto
    frame.delete("seghetto_lazo")
    tile_refs_seghetto = []
    for cx, cy, angolo in tile_info_seghetto:
        tile = get_rope_tile_seghetto(angolo)
        frame.create_image(cx, cy, image=tile, tags="seghetto_lazo")
        tile_refs_seghetto.append(tile)


def ottieni_area_seghetto():
    """Ritaglia l'area selezionata dal seghetto"""
    if not punti_seghetto:
        return None
    xs = [p[0] for p in punti_seghetto]
    ys = [p[1] for p in punti_seghetto]
    x_min, x_max = max(0, min(xs)), min(canvas_w, max(xs))
    y_min, y_max = max(0, min(ys)), min(canvas_h, max(ys))

    if x_min >= x_max or y_min >= y_max:
        return None

    return img_attuale.crop((x_min, y_min, x_max, y_max))



 
from tkinter import messagebox

conta_frames1 = 1
conta_framesch1 = 1
conta_framesch2 = 1
conta_framesch3 = 1
conta_framesch4 = 1
personaggio_in_uso = 1

AMPOLLA_FRAME_MASSIMO = 280
SOGLIA_MORTE_PERCENTO = 0.70
FRAME_TOTALI_CH = 80
SOGLIA_MORTE = round(SOGLIA_MORTE_PERCENTO * FRAME_TOTALI_CH)  # 24



OFFSET = 2   # ✅ rapporto tra velocità ampolla e (metà) velocità vita personaggio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATION_DIR = os.path.join(BASE_DIR, "location")

def calcola_blood():
    global imgbs, imgbsch, conta_frames1, window,immortalita
    global conta_framesch1, conta_framesch2, conta_framesch3, conta_framesch4
    global blood, blood_ch1, blood_ch2, blood_ch3, blood_ch4
    global personaggio_in_uso, oggetto_selezionato, Livel, save_livel

    # ------------------------------------------------------------
    # calcola i pixel del taglio/colpo in base all'oggetto usato
    # ------------------------------------------------------------
    if oggetto_selezionato is None:
        return  # nessun oggetto in mano, niente da calcolare

    nome_oggetto = os.path.basename(oggetto_selezionato).split('.')[0]

    if "cacciavite" in nome_oggetto:
        calcola_pixels_t = (128 * 123)
    elif "martello" in nome_oggetto:
        calcola_pixels_t = (128 * 82)
    elif "sedia" in nome_oggetto:
        return
    else:
        if not os.path.exists('seghetto_area.png'):
            return
        with Image.open("seghetto_area.png") as imgt:
            w, h = imgt.size
        calcola_pixels_t = (w * h)

    # ------------------------------------------------------------
    # quanto sangue viene tolto con QUESTO colpo/taglio
    # (fix: k_valido2 ora si aggiorna PRIMA del break, altrimenti
    #  restava sempre un passo indietro rispetto al vero valore)
    # ------------------------------------------------------------
    perdita_sangue = 0
    k_valido2 = 0
    for k in range(1, 1000):
        perdita_sangue += (30 * k)
        k_valido2 = k
        if calcola_pixels_t <= perdita_sangue:
            break

    if k_valido2 == 0:
        return  # colpo troppo debole, nessun effetto

    print(f"frames correnti prelevati: {k_valido2}")

    decremento_vita = k_valido2 // 2                 # solo metà del colpo va sulla vita
    incremento_ampolla = decremento_vita * OFFSET     # ampolla riceve il doppio di quella metà

    # ------------------------------------------------------------
    # aggiorna il contatore vita del personaggio giusto
    # ------------------------------------------------------------
    contatori = {1: conta_framesch1, 2: conta_framesch2, 3: conta_framesch3, 4: conta_framesch4}
    canvas_per_personaggio = {1: blood_ch1, 2: blood_ch2, 3: blood_ch3, 4: blood_ch4}

    valore_precedente = contatori[personaggio_in_uso]
    nuovo_valore = min(valore_precedente + decremento_vita, FRAME_TOTALI_CH - 1)

    if personaggio_in_uso == 1:
        conta_framesch1 = nuovo_valore
    elif personaggio_in_uso == 2:
        conta_framesch2 = nuovo_valore
    elif personaggio_in_uso == 3:
        conta_framesch3 = nuovo_valore
    elif personaggio_in_uso == 4:
        conta_framesch4 = nuovo_valore

    print(f"frames ch{personaggio_in_uso}: {nuovo_valore}")

    canvas_target = canvas_per_personaggio.get(personaggio_in_uso)
    if canvas_target is not None:
        imgbch = Image.open(imgbsch[nuovo_valore]).resize((80, 30), Image.BICUBIC)
        foto_ch = ImageTk.PhotoImage(imgbch)

        canvas_target.delete("blood_ch_frame")
        canvas_target.create_image(0, 0, anchor='nw', image=foto_ch, tags="blood_ch_frame")
        canvas_target.img_ch_photo = foto_ch
        canvas_target.update_idletasks()

        imgbch.close()

    # ------------------------------------------------------------
    # l'ampolla riceve il doppio della metà appena tolta alla vita
    # ------------------------------------------------------------
    conta_frames1 = min(conta_frames1 + incremento_ampolla, len(imgbs) - 1)
    print(f"frames ampolla: {conta_frames1}")

    imgb = Image.open(imgbs[conta_frames1]).resize((81, 281), Image.BICUBIC)
    foto_blood = ImageTk.PhotoImage(imgb)

    blood.delete("blood_frame")
    blood.create_image(0, 0, anchor='nw', image=foto_blood, tags="blood_frame")
    blood.img_blood_photo = foto_blood
    blood.update_idletasks()

    imgb.close()

    # ------------------------------------------------------------
    # controllo GAME OVER (controllato PRIMA della vittoria, per dare
    # priorità alla morte se scattano entrambe le condizioni nello stesso colpo)
    # ------------------------------------------------------------
    contatori_finali = {
        1: conta_framesch1,
        2: conta_framesch2,
        3: conta_framesch3,
        4: conta_framesch4,
    }
    for numero_char, valore in contatori_finali.items():
        if valore >= SOGLIA_MORTE and not immortalita:
            messagebox.showerror("Game Over", f"Personaggio {numero_char} morto")
            window.destroy()
            return  # fix: evita ulteriori messagebox/destroy se muore più di un personaggio

    # ------------------------------------------------------------
    # controllo VITTORIA / nuovo livello
    # ------------------------------------------------------------
    if conta_frames1 >= AMPOLLA_FRAME_MASSIMO:
        messagebox.showinfo("Livello completato", "Ampolla piena! Nuovo livello.")

        #avvia VIDEO ANIMAZIONE NEX LIVEL:
        video_path = os.path.join(BASE_DIR, "NextLivel.mp4")
        if os.path.exists(video_path):
            os.startfile(video_path)
        else:
            messagebox.showinfo("Errore", "Video Next Livello non trovato")
            # nota: NON facciamo return qui, altrimenti il livello
            # non avanza mai se manca solo il video
    

        for location_prec in ['attrezzi_flux', 'bagno_flux', 'camera da letto_flux',
                       'cucina_flux', 'parche_flux', 'portone_flux']:
            src = os.path.join(LOCATION_DIR, f"{location_prec}.png")
            dst = os.path.join(LOCATION_DIR, f"{location_prec}_{Livel}.png")
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"✅ Rinominato: {src} -> {dst}")
            else:
                print(f"⚠️ ATTENZIONE: file non trovato, impossibile rinominare: {src}")

        Livel += 1
        with open(save_livel, 'w') as f:
            f.write(str(Livel))

        # ------------------------------------------------------------
        # fix: RESET dei contatori vita/ampolla per il nuovo livello
        # ------------------------------------------------------------
        conta_frames1 = 1
        conta_framesch1 = 1
        conta_framesch2 = 1
        conta_framesch3 = 1
        conta_framesch4 = 1

        # ridisegna l'ampolla vuota
        imgb = Image.open(imgbs[conta_frames1]).resize((81, 281), Image.BICUBIC)
        foto_blood = ImageTk.PhotoImage(imgb)
        blood.delete("blood_frame")
        blood.create_image(0, 0, anchor='nw', image=foto_blood, tags="blood_frame")
        blood.img_blood_photo = foto_blood
        blood.update_idletasks()
        imgb.close()

        # ridisegna le barre vitali di tutti e 4 i personaggi
        canvas_per_personaggio = {1: blood_ch1, 2: blood_ch2, 3: blood_ch3, 4: blood_ch4}
        for num_ch, canvas_target in canvas_per_personaggio.items():
            imgbch = Image.open(imgbsch[1]).resize((80, 30), Image.BICUBIC)
            foto_ch = ImageTk.PhotoImage(imgbch)
            canvas_target.delete("blood_ch_frame")
            canvas_target.create_image(0, 0, anchor='nw', image=foto_ch, tags="blood_ch_frame")
            canvas_target.img_ch_photo = foto_ch
            canvas_target.update_idletasks()
            imgbch.close()

        def _prossimo_livello():
            genera_location_iniziali()
            time.sleep(1)

            # ------------------------------------------------------------
            # fix: elimina TUTTI i file temporanei prima del prossimo livello
            # ------------------------------------------------------------
            file_da_eliminare = ["cucina_flux1.png"]
            file_da_eliminare += [f"parte{k}_generata.png" for k in range(1, 5)]      # fix: range(1,5) -> 1,2,3,4
            file_da_eliminare += [f"sfondo_parte{k}.png" for k in range(1, 5)]        # fix: range(1,5) -> 1,2,3,4

            for nome_file in file_da_eliminare:
                percorso = os.path.join("history", nome_file)
                if os.path.exists(percorso):
                    os.remove(percorso)
                    print(f"🗑️ Eliminato: {percorso}")
                else:
                    print(f"⚠️ File non trovato, salto: {percorso}")

            avvia_gioco_thread()
            time.sleep(1)

        t.Thread(target=_prossimo_livello, daemon=True).start()

def f_lazo_seghetto_end(event):
    """Finisce il tracciamento del seghetto"""
    global punti_seghetto, tile_info_seghetto

    if not punti_seghetto:
        return

    print(f"🔪 Seghetto: tracciamento finito ({len(punti_seghetto)} punti)")

    frame.delete("seghetto_lazo")

    # Salva l'area se ha abbastanza punti
    if len(punti_seghetto) >= 5:
        area = ottieni_area_seghetto()
        if area:
            area.save("seghetto_area.png")
            print(f"✅ Area seghetto salvata: {area.size}")
            

    punti_seghetto = []
    tile_info_seghetto = []

# ============================================================
# NAVIGAZIONE STANZE
# ============================================================
ultimo_cambio_stanza = 0
DELAY_CAMBIO_STANZA = 1.0

def movie_mouse(event):
    global AVVIA_GIOCO
    global lab_stanza, path_stanza, ultimo_cambio_stanza
    global stanza_attuale, tk_img_attuale, img_attuale, oggetto, frame, oggetto_id, mouse_pos
    global testo

    # ✅ non fare nulla finché il gioco non è stato avviato
    if not AVVIA_GIOCO:
        return

    # ✅ PRIORITA': Se modalità oggetto è attiva, sposta l'oggetto
    if oggetto and oggetto_id is not None:
        mouse_pos = (event.x, event.y)
        frame.coords(oggetto_id, mouse_pos[0], mouse_pos[1])
        return

    # ✅ CONTROLLA il tempo prima di cambiare stanza
    tempo_attuale = time.time()
    if tempo_attuale - ultimo_cambio_stanza < DELAY_CAMBIO_STANZA:
        return
    print(f"x: {event.x},y:{event.y}")

    # ✅ ALTRIMENTI: Naviga la stanza normalmente
    direzione = None
    if event.x >= canvas_w - 5:
        direzione = "dx"
    elif event.x <= 5:
        direzione = "sx"
    elif event.y <= 5:
        direzione = "su"
    elif event.y >= canvas_h - 5:
        direzione = "giu"

    if direzione is None:
        return

    possibili = collegamenti.get(stanza_attuale, {})
    nuova_stanza = possibili.get(direzione)

    if nuova_stanza is None or nuova_stanza == stanza_attuale:
        return

    ultimo_cambio_stanza = tempo_attuale
    stanza_attuale = nuova_stanza

    # ✅ SE STANZA ATTUALE È CUCINA, RICOSTRUISCI IL COLLAGE
    if stanza_attuale == matrix_stanze[1][0]:
        print("🍳 Entrando in cucina... Ricostruisco collage")
        collage_path = ricostruisci_collage2(stanza_attuale)

        if collage_path and os.path.exists(collage_path):
            path_stanza = collage_path
            print(f"✅ Collage caricato: {path_stanza}")
        else:
            path_stanza = get_stanza_path(stanza_attuale)
            print(f"⚠️ Collage non disponibile, uso sfondo: {path_stanza}")
    else:
        path_stanza = get_stanza_path(stanza_attuale)

    # ✅ Carica e visualizza l'immagine
    try:
        img_attuale = Image.open(path_stanza).convert("RGB")
        img_attuale = img_attuale.resize((canvas_w, canvas_h))
        tk_img_attuale = ImageTk.PhotoImage(img_attuale)

        frame.delete("all")
        frame.create_image(0, 0, anchor='nw', image=tk_img_attuale)
        print(f"✅ Stanza Attuale: {path_stanza}")
        lab_stanza.config(text=f"Stanza: {os.path.basename(path_stanza)}")
        lab_stanza.update_idletasks()

        nome_base = os.path.basename(path_stanza).split('.')[0]

        p = None
        if 'attrezzi' in nome_base:
            p = """la ragazza con IL VESTITO ROSSO del image 2 è distessa in orizzontale sul tavolo degli attrezzi del image 1, nel inquadratura del tavolo sono entrambi visibili i piedi nudi e la testa; mantieni proporzioni del corpo della ragazza;
le braccia sono entrambi distesse lungo il corpo della ragazza verso i piedi, le gambe sono entrambi distesse sul tavolo. occhi aperti, sguardo assente.
Mantieni massima coerenza della ragazza nel image 2, del suo viso, dei suoi occhi, della sua capigliatura, del suo fisico, e del outfit."""
        elif 'parche' in nome_base:
            p = """la ragazza del image 2 con IL VESTITO ROSSO , è in piedi, con la (schiena dritta appoggiata alla colonna a destra del image 1).
la ragazza è posizionata in piedi appoggiata al palo destro del image 1 , con le braccia alzate in area appogiate al palo destro.
le mani socchiuse e vicine appoggiate al palo destro. 
posa non di profilo,posa della ragazza in piedi frontale.  
Mantieni la massima coerenza con la ragazza dell'image 2 per il viso, i capelli lisci, lunghi e castano scuro, gli occhi e la corporatura.
Mantieni l'outfit del vestito rosso corto con i dettagli a fasce incrociate sul busto.
Mantieni l'ambientazione del palcoscenico teatrale, delle tende rosse e dorate, della balaustra in legno scuro e del pavimento a parquet dell'image 1"""

        if p is not None:
            testo.delete('1.0', 'end')
            testo.insert('1.0', p)

    except Exception as e:
        print(f"❌ Errore caricamento immagine: {e}")


# ============================================================
# LAZO DELLA CORDA
# ============================================================

rope_img_originale = Image.open("lazo.png").convert("RGBA")
scala_corda = 0.35
nuova_w = int(rope_img_originale.width * scala_corda)
nuova_h = int(rope_img_originale.height * scala_corda)
rope_img_originale = rope_img_originale.resize((nuova_w, nuova_h), Image.LANCZOS)

pixel_data = rope_img_originale.get_flattened_data()
nuovi_pixel = [(r, g, b, 0) if r > 235 and g > 235 and b > 235 else (r, g, b, 255)
               for r, g, b, a in pixel_data]
rope_img_originale.putdata(nuovi_pixel)

rope_img_azzurra = tinteggia_corda(rope_img_originale, (80, 170, 255))
rope_w, rope_h = rope_img_originale.size

rope_cache_normale = {}
rope_cache_azzurra = {}

def get_rope_tile(angolo_gradi, azzurro=False):
    angolo_arrotondato = round(angolo_gradi / 5) * 5
    sorgente = rope_img_azzurra if azzurro else rope_img_originale
    cache = rope_cache_azzurra if azzurro else rope_cache_normale
    if angolo_arrotondato not in cache:
        ruotata = sorgente.rotate(-angolo_arrotondato, expand=True, resample=Image.BICUBIC)
        cache[angolo_arrotondato] = ImageTk.PhotoImage(ruotata)
    return cache[angolo_arrotondato]


# ============================================================
# STATO DEL LAZO
# ============================================================

lazo_attivo = False
punti_corda = []
tile_info = []          # (cx, cy, angolo) di ogni pezzo piazzato, per poterli ridisegnare
ultimo_tile_pos = None
tile_refs = []
lazo_chiuso = False

SOGLIA_CHIUSURA = 25         # px di tolleranza: quanto vicino al punto di partenza per considerarlo "chiuso"
MIN_PUNTI_PER_CHIUSURA = 8   # evita falsi positivi appena inizi a disegnare


def f_lazo_start(event):
    global punti_corda, ultimo_tile_pos, tile_refs, tile_info, lazo_chiuso
    if not lazo_attivo:
        return
    frame.delete("corda")
    tile_refs = []
    tile_info = []
    punti_corda = [(event.x, event.y)]
    ultimo_tile_pos = (event.x, event.y)
    lazo_chiuso = False


def f_draw_lazo(event):
    global ultimo_tile_pos, punti_corda, tile_refs, tile_info, lazo_chiuso
    if not lazo_attivo or not punti_corda:
        return

    x0, y0 = ultimo_tile_pos
    dx, dy = event.x - x0, event.y - y0
    distanza = math.hypot(dx, dy)
    spaziatura = rope_w * 0.75

    if distanza >= spaziatura:
        angolo = math.degrees(math.atan2(dy, dx))
        cx, cy = x0 + dx / 2, y0 + dy / 2

        tile_info.append((cx, cy, angolo))
        tile = get_rope_tile(angolo, azzurro=lazo_chiuso)
        frame.create_image(cx, cy, image=tile, tags="corda")
        tile_refs.append(tile)

        ultimo_tile_pos = (event.x, event.y)
        punti_corda.append((event.x, event.y))

    # --- Controlla se il cappio si è appena chiuso o riaperto ---
    x_iniz, y_iniz = punti_corda[0]
    distanza_da_inizio = math.hypot(event.x - x_iniz, event.y - y_iniz)

    era_chiuso = lazo_chiuso
    lazo_chiuso = (len(punti_corda) >= MIN_PUNTI_PER_CHIUSURA
                   and distanza_da_inizio <= SOGLIA_CHIUSURA)

    if lazo_chiuso != era_chiuso:
        ridisegna_lazo()
        if lazo_chiuso:
            avvia_riconoscimento_personaggio()


def ridisegna_lazo():
    global tile_refs
    frame.delete("corda")
    tile_refs = []
    for cx, cy, angolo in tile_info:
        tile = get_rope_tile(angolo, azzurro=lazo_chiuso)
        frame.create_image(cx, cy, image=tile, tags="corda")
        tile_refs.append(tile)


def ottieni_area_lazo():
    """Ritaglia dall'immagine della stanza attuale il rettangolo che racchiude il cappio"""
    xs = [p[0] for p in punti_corda]
    ys = [p[1] for p in punti_corda]
    x_min, x_max = max(0, min(xs)), min(canvas_w, max(xs))
    y_min, y_max = max(0, min(ys)), min(canvas_h, max(ys))
    return img_attuale.crop((x_min, y_min, x_max, y_max))


def f_lazo_end(event):
    global punti_corda, tile_info, lazo_chiuso
    frame.delete("corda")
    punti_corda = []
    tile_info = []
    lazo_chiuso = False


# ============================================================
# RICONOSCIMENTO PERSONAGGIO (lazo corda)
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import re

selected_char = None

def avvia_riconoscimento_personaggio():
    """Riconoscimento volti con MediaPipe"""
    global selected_char, select_body_var, select_face_var

    ritaglio = ottieni_area_lazo()

    if ritaglio.width < 5 or ritaglio.height < 5:
        print("Area troppo piccola, ignorata")
        return

    ritaglio.save("area.png")
    time.sleep(1)

    # Carica immagine con OpenCV
    image_nota = cv2.imread("area.png")

    if image_nota is None:
        print("Errore: impossibile leggere area.png")
        return

    image_nota_rgb = cv2.cvtColor(image_nota, cv2.COLOR_BGR2RGB)

    # Usa MediaPipe per il riconoscimento
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection() as face_detection:
        results_nota = face_detection.process(image_nota_rgb)

        if not results_nota.detections:
            print("Nessun volto trovato in area.png")
            return

        characters = ['./character/ch1face.png', './character/ch2face.png',
                     './character/ch3face.png', './character/ch4face.png']

        for f in characters:
            image = cv2.imread(f)

            if image is None:
                print(f"Errore: impossibile leggere {f}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_char = face_detection.process(image_rgb)

            if results_char.detections:
                # Estrai il numero dal path (es. ch1face.png -> 1)
                match = re.search(r'ch(\d+)', f)
                if match:
                    char_num = int(match.group(1))  # es. 1, 2, 3, 4
                    char_index = char_num - 1  # converte a indice 0-3

                    # Seleziona full o face basato sul checkbox
                    if select_body_var.get():
                        selected_char = f.replace("face", "full")
                    else:
                        selected_char = f

                    print(f"Match trovato: {selected_char}")

                    # ✅ AGGIORNA LA UI AUTOMATICAMENTE
                    # Chiama toggle_select con il frame_canvas e l'indice corretti
                    toggle_select(char_canvases[char_index].master, char_index)

                break


def motion_handler(event):
    """Gestisce movimento mouse: lazo corda OR seghetto OR pennello OR navigazione"""
    global Punti_disegno, frame, diametro_corrente, pressione_corrente
    global disegna_attivo, cancella_attivo, colore_corrente

    if lazo_attivo:
        f_draw_lazo(event)
        return

    if seghetto_attivo_ora():
        f_draw_lazo_seghetto(event)
        return

    if disegna_attivo or cancella_attivo:
        Punti_disegno.append((event.x, event.y))
        img, draw = get_image_disegno()

        if len(Punti_disegno) >= 2:
            x0, y0 = Punti_disegno[-2]
            x1, y1 = Punti_disegno[-1]

            fattore_pressione = pressione_corrente / 100.0
            spessore = max(1, int(diametro_corrente * fattore_pressione))

            if cancella_attivo:
                raggio = spessore / 2
                elementi_vicini = frame.find_overlapping(
                    x1 - raggio, y1 - raggio, x1 + raggio, y1 + raggio
                )
                for item in elementi_vicini:
                    if TAG_DISEGNO in frame.gettags(item):
                        frame.delete(item)
                draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0, 0), width=spessore)
            else:
                frame.create_line(
                    x0, y0, x1, y1,
                    fill=colore_corrente,
                    width=spessore,
                    capstyle=tk.ROUND,
                    smooth=True,
                    tags=TAG_DISEGNO
                )
                frame.tag_raise(TAG_DISEGNO)
                draw.line([(x0, y0), (x1, y1)], fill=colore_corrente, width=spessore)
        return

    movie_mouse(event)


def on_press(event):
    global Punti_disegno, disegna_attivo

    if lazo_attivo:
        f_lazo_start(event)
    elif seghetto_attivo_ora():
        f_lazo_seghetto_start(event)
    elif disegna_attivo:
        Punti_disegno = [(event.x, event.y)]


def on_release(event):
    global Punti_disegno, image_disegno, disegna_attivo, cancella_attivo

    if lazo_attivo:
        f_lazo_end(event)
    elif seghetto_attivo_ora():
        f_lazo_seghetto_end(event)
    elif disegna_attivo or cancella_attivo:
        if image_disegno is not None:
            image_disegno.save("disegno.png")
        Punti_disegno = []


frame.bind("<ButtonPress-1>", on_press)
frame.bind("<ButtonRelease-1>", on_release)
frame.bind("<B1-Motion>", motion_handler)

frame_testo = tk.Frame(frame_centro, bg='gray')
frame_testo.grid(row=1, column=0, sticky='ew', pady=(2, 0))
frame_testo.columnconfigure(0, weight=1)

testo = tk.Text(frame_testo, width=80, height=5)
testo.grid(row=0, column=0, sticky='ew')

frame_lab = tk.Frame(frame_centro)
frame_lab.grid(row=2, column=0, sticky='n', pady=5)

lab_stanza = tk.Label(frame_lab, text='Stanza: Cucina')
lab_stanza.grid(row=0, column=0)

lab_character = tk.Label(frame_lab, text='Character: nessuno')
lab_character.grid(row=0, column=1)

lab_oggetto = tk.Label(frame_lab, text='Oggetto: nessuno')
lab_oggetto.grid(row=0, column=2)

# ============================================================
# COLONNA 2 — Strumenti + Personaggi
# ============================================================

button_game = tk.Frame(window, bg='gray')
button_game.grid(row=0, column=2, sticky='n', padx=(0, 10), pady=10)

lazo_attivo = False
punti_corda = []  # lista di tutti i punti (x, y) per disegnare la corda

def f_lazo():
    global lazo_attivo
    lazo_attivo = not lazo_attivo
    if lazo_attivo:
        lazo_btn.config(bg='light green')
    else:
        lazo_btn.config(bg='light yellow')

lazo_btn = tk.Button(button_game, text='Lazo', width=10, bg='light yellow', command=f_lazo)
lazo_btn.grid(row=0, column=0, pady=5, padx=5)

apri_pennello = False
frame_pennello = None
disegna_attivo = False
cancella_attivo = False
diametro_corrente = 1.0
pressione_corrente = 1.0
colore_corrente = 'black'

from tkinter import colorchooser
from PIL import Image, ImageDraw

Punti_disegno = []
TAG_DISEGNO = 'disegno'
image_disegno = None
draw_disegno = None
new_color = None

def get_image_disegno():
    """Crea (una sola volta) l'immagine trasparente delle stesse dimensioni del canvas."""
    global image_disegno, draw_disegno
    if image_disegno is None:
        larghezza = frame.winfo_width()
        altezza = frame.winfo_height()
        image_disegno = Image.new("RGBA", (larghezza, altezza), (0, 0, 0, 0))
        draw_disegno = ImageDraw.Draw(image_disegno)
    return image_disegno, draw_disegno


def f_frame_pennello():
    global apri_pennello, frame_pennello, disegna_attivo, cancella_attivo, diametro_corrente, pressione_corrente, new_color, colore_corrente
    if not apri_pennello:
        if frame_pennello is None or not frame_pennello.winfo_exists():
            frame_pennello = tk.Toplevel()
            frame_pennello.geometry("400x600")
            frame_pennello.resizable(False, False)
            frame_pennello.config(background='gray')
            frame_pennello.lift()

            frame_pennello.update_idletasks()
            x = frame_pennello.winfo_screenwidth() - frame_pennello.winfo_width()
            y = 0
            frame_pennello.geometry(f"+{x}+{y}")

            def f_dis():
                global disegna_attivo, cancella_attivo
                if disegna_attivo:
                    disegna_attivo = False
                else:
                    disegna_attivo = True
                    cancella_attivo = False

            def f_canc():
                global disegna_attivo, cancella_attivo
                if cancella_attivo:
                    cancella_attivo = False
                else:
                    cancella_attivo = True
                    disegna_attivo = True

            btn_disegna = tk.Button(frame_pennello, text='Disegna', bg='green', command=f_dis)
            btn_disegna.grid(row=0, column=0, pady=10, padx=5, sticky='w')

            btn_gomma = tk.Button(frame_pennello, text='Gomma', bg='#ff8888', command=f_canc)
            btn_gomma.grid(row=1, column=0, pady=10, padx=5, sticky='w')

            def resetta():
                global frame, Punti_disegno, image_disegno, draw_disegno
                frame.delete(TAG_DISEGNO)
                if image_disegno is not None:
                    larghezza, altezza = image_disegno.size
                    image_disegno = Image.new("RGBA", (larghezza, altezza), (0, 0, 0, 0))
                    draw_disegno = ImageDraw.Draw(image_disegno)
                if os.path.exists('./disegno.png'):
                    os.remove('./disegno.png')
                Punti_disegno = []

            btn_cancella = tk.Button(frame_pennello, text='Cancella', bg='light blue', command=resetta)
            btn_cancella.grid(row=2, column=0, pady=10, padx=5, sticky='w')

            pressione_valore = tk.DoubleVar(value=pressione_corrente)
            lab_pressione_penna = tk.Label(frame_pennello, text=f'Pressione Penna: {pressione_valore.get():.1f}')
            lab_pressione_penna.grid(row=3, column=0, pady=(20, 0), padx=5, sticky='w')

            def aggiorna_pressione(val):
                global pressione_corrente
                pressione_corrente = float(val)
                lab_pressione_penna.config(text=f'Pressione Penna: {pressione_corrente:.1f}')
                pressione_valore.set(pressione_corrente)

            pressione_penna = ttk.Scale(frame_pennello, from_=0.1, to=100.0, orient=HORIZONTAL,
                                        variable=pressione_valore, command=aggiorna_pressione)
            pressione_penna.grid(row=4, column=0, pady=5, padx=5, sticky='ew')
            pressione_penna.set(20.0)

            diametro_valore = tk.DoubleVar(value=diametro_corrente)
            lab_diametro_penna = tk.Label(frame_pennello, text=f'Diametro Penna: {diametro_valore.get():.1f}')
            lab_diametro_penna.grid(row=5, column=0, pady=(20, 0), padx=5, sticky='w')

            def aggiorna_diametro(val):
                global diametro_corrente
                diametro_corrente = float(val)
                lab_diametro_penna.config(text=f'Diametro Penna: {diametro_corrente:.1f}')
                diametro_valore.set(diametro_corrente)

            diametro_penna = ttk.Scale(frame_pennello, from_=0.1, to=100.0, orient=HORIZONTAL,
                                       variable=diametro_valore, command=aggiorna_diametro)
            diametro_penna.grid(row=6, column=0, pady=5, padx=5, sticky='ew')
            diametro_penna.set(20.0)

            def f_color(event=None):
                global new_color, colore_corrente
                rgb, esadecimale = colorchooser.askcolor(color=colore_corrente, parent=frame_pennello)
                if esadecimale is not None:
                    colore_corrente = esadecimale
                    new_color.config(bg=colore_corrente)

            new_color = tk.Canvas(frame_pennello, bg=colore_corrente, width=30, height=30)
            new_color.grid(row=7, column=0)
            new_color.bind('<Button-1>', f_color)

            def on_close():
                global apri_pennello
                apri_pennello = False
                frame_pennello.destroy()
            frame_pennello.protocol("WM_DELETE_WINDOW", on_close)
        apri_pennello = True
    else:
        if frame_pennello is not None and frame_pennello.winfo_exists():
            frame_pennello.destroy()
        apri_pennello = False


pennello = tk.Button(
    button_game,
    text='Pennello',
    width=10,
    bg='light blue',
    command=lambda: t.Thread(target=f_frame_pennello, daemon=True).start()
)
pennello.grid(row=1, column=0, pady=5, padx=5)

max_indice = 1
if os.path.exists("indice.txt"):
    with open("indice.txt", 'r') as f:
        max_indice = int(f.read())
    var_indice.set(max_indice)  # ✅ Usa var_indice
    spinbox_indice.update_idletasks()

def ricostruisci_collage2(stanza_nome, out_dir="./history"):
    """
    Riaffianca orizzontalmente le 4 immagini generate (una per ogni personaggio)
    per ricreare l'immagine finale completa.
    Se il personaggio è stato usato (True) → NON c'è in cucina → usa sfondo_parte*
    Se il personaggio NON è stato usato (False) → C'è in cucina → usa parte*_generata
    """
    
    # ✅ Leggi quali personaggi sono stati usati
    use_characters = leggi_character_file()
    
    # ✅ Costruisci i path in base agli usi (LOGICA INVERTITA)
    paths_generati = []
    
    # Parte 1
    if use_characters["ch1"]:
        # ✅ SE USATO (True) → non c'è in cucina → usa sfondo
        path = os.path.join(out_dir, "sfondo_parte1.png")
    else:
        # ✅ SE NON USATO (False) → c'è in cucina → usa generata
        path = os.path.join(out_dir, "parte1_generata.png")
    
    if os.path.exists(path):
        paths_generati.append(path)
        print(f"✅ Parte1: {path}")
    else:
        print(f"❌ Parte1 non trovata: {path}")
        return None
    
    # Parte 2
    if use_characters["ch2"]:
        path = os.path.join(out_dir, "sfondo_parte2.png")
    else:
        path = os.path.join(out_dir, "parte2_generata.png")
    
    if os.path.exists(path):
        paths_generati.append(path)
        print(f"✅ Parte2: {path}")
    else:
        print(f"❌ Parte2 non trovata: {path}")
        return None
    
    # Parte 3
    if use_characters["ch3"]:
        path = os.path.join(out_dir, "sfondo_parte3.png")
    else:
        path = os.path.join(out_dir, "parte3_generata.png")
    
    if os.path.exists(path):
        paths_generati.append(path)
        print(f"✅ Parte3: {path}")
    else:
        print(f"❌ Parte3 non trovata: {path}")
        return None
    
    # Parte 4
    if use_characters["ch4"]:
        path = os.path.join(out_dir, "sfondo_parte4.png")
    else:
        path = os.path.join(out_dir, "parte4_generata.png")
    
    if os.path.exists(path):
        paths_generati.append(path)
        print(f"✅ Parte4: {path}")
    else:
        print(f"❌ Parte4 non trovata: {path}")
        return None
    
    # ✅ Carica le immagini
    try:
        immagini = [Image.open(p).convert("RGB") for p in paths_generati]
    except Exception as e:
        print(f"❌ Errore caricamento immagini: {e}")
        return None

    h = immagini[0].height
    w_totale = sum(im.width for im in immagini)

    # ✅ Crea l'immagine finale
    finale = Image.new("RGB", (w_totale, h), (0, 0, 0))
    x_offset = 0
    
    for im in immagini:
        if im.height != h:
            ratio = h / im.height
            im = im.resize((int(im.width * ratio), h), Image.LANCZOS)
        finale.paste(im, (x_offset, 0))
        x_offset += im.width

    # ✅ Salva con il nome corretto
    os.makedirs(out_dir, exist_ok=True)
    nome_finale = increment_filename(f"{stanza_nome}.png", out_dir)
    out_path = os.path.join(out_dir, f"{nome_finale}.png")
    
    finale.save(out_path)
    print(f"✅ Immagine finale ricomposta salvata in: {out_path}")
    return out_path

char_path = None
import re
import os
import glob
import random

def increment_filename(path, out_dir):
    """
    Incrementa il numero alla fine del filename fino a trovare un nome disponibile
    cucina.png esiste → salva come cucina1.png
    cucina1.png esiste → salva come cucina2.png
    etc.
    """
    nome_base = os.path.basename(path).split('.')[0]
    estensione = os.path.basename(path).split('.')[-1]
    
    # Rimuovi il numero alla fine del nome base (se esiste)
    nome_pulito = re.sub(r'_?\d+$', '', nome_base)
    
    # Controlla se esiste già il file senza numero
    counter = 1
    while True:
        # Tentativi successivi: aggiungi numero (es. "cucina1.png", "cucina2.png")
        filename_completo = f"{nome_pulito}{counter}.{estensione}"
        filepath_completo = os.path.join(out_dir, filename_completo)
        
        # Se il file non esiste, usa questo nome
        if not os.path.exists(filepath_completo):
            print(f"✅ Nome disponibile: {filename_completo}")
            return filename_completo.split('.')[0]  # Ritorna solo il nome senza estensione
        
        counter += 1
        print(f"⚠️ {filename_completo} esiste, provo il prossimo...")


# ✅ Funzione per leggere i personaggi usati
def leggi_character_file():
    """Legge quale personaggio è stato usato"""
    use_characters = {"ch1": False, "ch2": False, "ch3": False, "ch4": False}
    
    if os.path.exists("use_character.txt"):
        try:
            with open("use_character.txt", 'r') as f:
                for line in f:
                    line = line.strip()
                    if "ch1:" in line:
                        use_characters["ch1"] = line.split("ch1:")[1].lower() == "true"
                    elif "ch2:" in line:
                        use_characters["ch2"] = line.split("ch2:")[1].lower() == "true"
                    elif "ch3:" in line:
                        use_characters["ch3"] = line.split("ch3:")[1].lower() == "true"
                    elif "ch4:" in line:
                        use_characters["ch4"] = line.split("ch4:")[1].lower() == "true"
        except Exception as e:
            print(f"⚠️ Errore lettura character file: {e}")
    
    return use_characters


# ✅ Funzione per scrivere i personaggi usati
def scrivi_character_file(use_characters):
    """Scrive quale personaggio è stato usato"""
    try:
        with open("use_character.txt", 'w') as f:
            f.write(f"ch1:{str(use_characters['ch1']).lower()}\n")
            f.write(f"ch2:{str(use_characters['ch2']).lower()}\n")
            f.write(f"ch3:{str(use_characters['ch3']).lower()}\n")
            f.write(f"ch4:{str(use_characters['ch4']).lower()}\n")
        print("✅ Character file salvato")
    except Exception as e:
        print(f"❌ Errore scrittura character file: {e}")


# GENERA IMMAGINE 
# GENERA IMMAGINE 
# GENERA IMMAGINE 

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# GENERA IMMAGINE
# ----------------------------------------------------------------------
def genera_image():
    print("genera nuova immagine")
    global path_stanza, oggetto_selezionato, char_path, spinbox_indice, testo, steps, lora, canvas_w, canvas_h
    global frame, max_indice, stanza_attuale, tk_img_attuale, img_attuale, var_indice
 
    try:
        # Validazione input
        prompt = testo.get('1.0', tk.END).strip()
        if not prompt:
            print("Prompt vuoto!")
            return
 
        try:
            steps_value = int(steps.get())
            if steps_value <= 0:
                raise ValueError("Steps deve essere > 0")
        except ValueError as e:
            print(f"Errore steps: {e}")
            return
 
        # ------------------------------------------------------------------
        # LOGICA PATH - path1 (sfondo), path2 (elemento principale), path3 (ferita se serve)
        # ------------------------------------------------------------------
        path_stanza=get_stanza_path(stanza_attuale)
        print(f"path_stanza: {path_stanza}")
        path1 = path_stanza if os.path.exists(path_stanza) else None
        path2 = None
        path3 = None
 
        # Determina se l'oggetto è sedia
        is_sedia = False
        nome_oggetto = None
        if oggetto_selezionato:
            nome_oggetto = os.path.basename(oggetto_selezionato).split('.')[0].lower()
            is_sedia = "sedia" in nome_oggetto
 
        if is_sedia:
            # SEDIA: priorità personaggio → sedia (nessuna ferita)
            if char_path and os.path.exists(char_path):
                path2 = char_path
            elif oggetto_selezionato and os.path.exists(oggetto_selezionato):
                path2 = oggetto_selezionato
            # else: path2 rimane None
 
        else:
            # ALTRI OGGETTI (martello, cacciavite, seghetto)
            if oggetto_selezionato and os.path.exists(oggetto_selezionato):
                
                path2 =f'./ferite//ferita ({random.randint(1,7)}).png'
 
                # caso speciale cacciavite: se il prompt gia' in textbox chiede
                # occhi mancanti/orbite, la ferita va presa dalla cartella occhi
                if nome_oggetto == 'cacciavite':
                    if "occhi mancanti" in prompt or "orbite" in prompt or "senza occhi" in prompt:
                        path2 = f'./ferite/occhi ({random.randint(1, 2)}).png'
 
                path3 = oggetto_selezionato
 
            elif char_path and os.path.exists(char_path):
                # nessun oggetto: path2 = personaggio, nessuna ferita/oggetto
                path2 = char_path
                path3 = None
            # else: path2 e path3 rimangono None
 
        print(f"path1 (sfondo): {path1}")
        print(f"path2 (ferita/personaggio): {path2}")
        print(f"path3 (oggetto): {path3}")
        print(f"oggetto: {nome_oggetto} (sedia={is_sedia})")
 
        # Validazione LoRA
        lora_path = f"./lora/{lora.get()}.safetensors"
        lora_path_verificato = lora_path if os.path.exists(lora_path) else None
 
        if not lora_path_verificato:
            print(f"Warning: LoRA file non trovato: {lora_path}")
 
        # Setup directory
        history_dir = os.path.abspath("history")
        os.makedirs(history_dir, exist_ok=True)
        print(f"Directory salvataggio: {history_dir}")
 
        # Genera immagine
        name = increment_filename(path_stanza, history_dir)
        flux2(
            prompt=prompt,
            steps=steps_value,
            path_image1=path1,
            path_image2=path2,
            path_image3=path3,
            path_lora=lora_path_verificato,
            wc=canvas_w,
            hc=canvas_h,
            out_dir=history_dir,
            name=name
        )
 
        # Aggiorna traccia character
        use_characters = leggi_character_file()
        if char_path:
            for i in range(1, 5):
                if str(i) in char_path or f'ch{i}' in char_path:
                    use_characters[f"ch{i}"] = True
 
        print("Character utilizzati:")
        for ch, usado in use_characters.items():
            print(f"  {ch}: {usado}")
        scrivi_character_file(use_characters)
 
        # Aggiorna indice
        var_indice.set(var_indice.get() + 1)
 
        if var_indice.get() > max_indice:
            max_indice = var_indice.get()
            with open("indice.txt", 'w') as fh:
                fh.write(str(max_indice))
            print(f"Indice aggiornato: {max_indice}")
 
        # Callback per aggiornare canvas (thread-safe)
        frame.after(0, calcola_blood)
        frame.after(100, aggiorna_canvas)
 
    except Exception as e:
        print(f"Errore in genera_image: {e}")
        import traceback
        traceback.print_exc()
 
 
def aggiorna_canvas():
    """Aggiorna la canvas con la nuova immagine (eseguita nel thread principale)"""
    global path_stanza, img_attuale, tk_img_attuale
 
    try:
        print("Caricamento immagine generata nella canvas...")
        path_stanza = get_stanza_path(stanza_attuale)
 
        if not os.path.exists(path_stanza):
            print(f"Immagine non trovata: {path_stanza}")
            return
 
        img_attuale = Image.open(path_stanza).convert("RGB")
        img_attuale = img_attuale.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        tk_img_attuale = ImageTk.PhotoImage(img_attuale)
 
        frame.delete("all")
        frame.create_image(0, 0, anchor='nw', image=tk_img_attuale)
        print(f"Canvas aggiornata: {os.path.basename(path_stanza)}")
 
        lab_stanza.config(text=f"Stanza: {os.path.basename(path_stanza)}")
        lab_stanza.update_idletasks()
 
    except Exception as e:
        print(f"Errore aggiornamento canvas: {e}")
    finally:
        spinbox_indice.update_idletasks()
        frame.update_idletasks()
 
 
genera = tk.Button(
    button_game,
    text='Genera',
    width=10,
    bg='light green',
    command=lambda: t.Thread(target=genera_image, daemon=True).start()
)
genera.grid(row=2, column=0, pady=5, padx=5)


lab_steps = tk.Label(button_game, text='Step 8', bg='gray', fg='white')
lab_steps.grid(row=3, column=0, pady=5)


def f_val(val):
    lab_steps.config(text=f"Step {int(float(val))}")


steps = ttk.Scale(button_game, from_=1, to=50, orient=HORIZONTAL, command=f_val)
steps.set(8)
steps.grid(row=4, column=0, pady=5, padx=5)

lab_lora = tk.Label(button_game, text='Lora')
lab_lora.grid(row=5, column=0, pady=5, padx=5)


def load_lora():
    global lora
    try:
        lora_files = [os.path.basename(l).split('.')[0] 
                     for l in os.listdir("lora") 
                     if os.path.isfile(os.path.join("lora", l))]
        
        # Clean list
        values = ["no_lora"]  + lora_files
        # Remove duplicates while preserving order
        seen = set()
        values = [x for x in values if not (x in seen or seen.add(x))]
        
        lora['values'] = values
        lora.update_idletasks()
    except FileNotFoundError:
        print("Cartella 'lora' non trovata!")
        lora['values'] = ["no_lora"]

def select_lora(event=None):
    global path_stanza, testo, lora
    
    selected_lora = lora.get().strip()
    if not selected_lora or selected_lora == "no_lora":
        testo.delete('1.0', tk.END)
        testo.insert('1.0', "Nessuna LoRA selezionata.")
        return

    nome_stanza = os.path.basename(get_stanza_path(path_stanza)).split('.')[0].lower()
    
    # ====================== DETECTION STANZA ======================
    luogo_azione = "letto"  # default
    if 'attrezzi' in nome_stanza:
        luogo_azione = "tavolo degli attrezzi"
        location_desc = "in una stanza degli attrezzi tetra e abbandonata, pareti di cemento sporche, tavolo di legno insanguinato, attrezzi arrugginiti appesi al muro, illuminazione fredda e drammatica da lampadina nuda"
    elif 'bagno' in nome_stanza:
        luogo_azione = "wc,tazza"
        location_desc = "in un bagno abbandonato e macabro, pareti con carta da parati strappata e macchiata di sangue, pavimento sporco con pozzanghere rosse, vasca da bagno vecchia, toilette rotta, atmosfera horror notturna"
    elif any(x in nome_stanza for x in ['camera', 'letto', 'bedroom']):
        luogo_azione = "letto matrimoniale"
        location_desc = "in una camera da letto vecchia e inquietante illuminata da luce rossa calda, letto disfatto con lenzuola rosse, mobili di legno antichi, atmosfera intima e disturbante"
    elif any(x in nome_stanza for x in ['parche', 'teatro', 'stage']):
        luogo_azione = "sedia del palco del teatro"
        location_desc = "su un palco di un teatro antico lussuoso e decadente, tende rosse pesanti, luci teatrali drammatiche, balconate di legno scuro, atmosfera elegante ma inquietante"
    else:
        location_desc = "nella stessa ambientazione della reference image 1"

    _COERENZA = "Mantieni massima coerenza della ragazza con l'image 2 (viso, capelli, occhi, corporatura) e massima coerenza dell'ambientazione dell'image 1"

    prompt = f"{selected_lora}, "

    # ====================== PROMPT PER LORA ======================

    if 'clothesonoffv2' in selected_lora:
        prompt += f"""ragazza completamente nuda, vista frontale, corpo esposto in modo esplicito, {location_desc}, 
seno naturale, capezzoli dettagliati, pube con leggera peluria naturale, dettagli genitali femminili realistici, {_COERENZA}"""

# ====================== SPREAD LEGS ======================
    elif 'spread_legs_beta1cum' in selected_lora:
        prompt += f"""spread_legs_beta1, fotorealistica, donna identica al image 2, massima coerenza del viso, capelli, occhi, 
sdraiata supina sul {luogo_azione}, gambe sollevate e divaricate in aria, mani che tengono le cosce aperte, 
seni naturali nudi, capezzoli dettagliati, fisico snello e magro,
inquadratura ravvicinata sulla vagina estremamente dilatata, messa a fuoco su viso e figa,
interno della figa completamente riempito di sperma spesso, vulva stracolma di sborra bianca che cola abbondantemente verso l'ano, 
clitoride gonfio e visibile, pube con pochi peli castani sottili, 
grandi labbra della figa molto aperte e tirate verso l'esterno, piccole labbra estremamente divaricate e tirate, dettagliate e bagnate, 
sperma bianco traslucido che riempie la vagina e cola verso l'ano,
ano estremamente dilatato con rughe realistiche e texture dettagliata della pelle,
{location_desc}, alta risoluzione, photorealistic, pelle realistica, illuminazione morbida, dettagli estremi, {_COERENZA}"""

    elif 'spread_legs_beta1' in selected_lora:
        prompt += f"""spread_legs_beta1, ragazza fotorealistica, stesso volto del image 2, completamente nuda, 
distesa supina sul {luogo_azione}, gambe estremamente divaricate e sollevate in aria, ginocchia piegate, piedi visibili in telecamera, 
mani che tirano i glutei per aprire al massimo,
seno naturale morbido, capezzoli realistici, leggera peluria pubica brunetta, (figa estremamente dilatata), 
(grandi labbra della vagina estremamente dilatate), (piccole labbra della vagina estremamente dilatate), 
vulva aperta e bagnata, canale vaginale aperto con interno visibile, clitoride visibile, 
ano estremamente dilatato, enorme apertura anale rilassata, bordo anale stirato al massimo, profondità rettale visibile, 
pelle iperrealistica con texture dettagliata, lucida di succhi, {location_desc}, 
foto raw 8k, qualità massima, dettaglio anatomico estremo, realismo fotografico crudo, {_COERENZA}"""


    # ====================== DILDO ======================
    elif 'dildo1' in selected_lora:
        prompt += f"""anus_insertion_v1, Fotografia di una donna dell'image 2, totalmente nuda, capelli lunghi neri, seno naturale, capezzoli realistici visibili, 
distesa su {luogo_azione} con le gambe alzate e divaricate, guarda verso la fotocamera, {location_desc}, 
Dall'ano spunta un grosso oggetto nero, poca peluria pubica castana naturale, vulva bagnata estremamente dettagliata, 
grandi labbra carnose, piccole labbra lunghe divaricate, profonda apertura vaginale, fluidi luccicanti, clitoride gonfio,
dettagli anali: ano estremamente dilatato, apertura visibile, righe intorno all'ano, texture pelle iperrealistica,
occhi luminosi, labbra carnose con rossetto rosso, foto grezza 8k, photorealistic, raw photo, {_COERENZA}"""

    elif 'dildo2' in selected_lora:
        prompt += f"""anus_insertion_v1, Fotografia della donna del image 2, completamente nuda, seno grande naturale, capezzoli realistici visibili, capelli lunghi, 
accovacciata sul {luogo_azione} con le gambe divaricate, pube rasato, dettagli genitali femminili, {location_desc}, 
di fronte alla fotocamera, sguardo verso l'obiettivo, grosso oggetto nero inserito nel suo ano, dilatazione ano estrema,
L'inquadratura è frontale, {_COERENZA}"""

    elif 'PornMaster_innie_pussy' in selected_lora:
        prompt += f"""PornMaster_innie_pussy, ripresa da dietro, ragazza del image 2 a quattro zampe, {location_desc}, 
ano e figa estremamente dilatati con grandi e piccole labbra aperte, innie pussy dettagliata, 
coerenza viso di profilo, corpo femminile neutro, {_COERENZA}"""

    # ==================== ORAL ====================
    elif 'POV_blowjobV1_A' in selected_lora:
        prompt += f"""POV_blowjobV1_A, inquadratura in prima persona di una ragazza del image 2 dalle labbra rosse mentre fa un pompino, 
{location_desc}, ripresa dal punto di vista dell'uomo, luce drammatica, stile spontaneo e realistico, {_COERENZA}"""

    elif 'blowjob_klein_v1_side' in selected_lora:
        prompt += f"""blowjob_klein_v1_side, ragazza del image 2 in ginocchio mentre pratica sesso orale a un uomo in piedi, penetrazione orale profonda visibile, 
mano dell'uomo sulla sua testa, {location_desc}, seno naturale visibile, espressione concentrata, {_COERENZA}"""

    elif 'FK_cuminmouth8' in selected_lora:
        prompt += f"""FK_cuminmouth8, ragazza del image 2 totalmente nuda, seno nudo naturale, capezzoli dettagliati, bocca aperta piena di sperma, 
sperma visibile sulle labbra e lingua, {location_desc}, espressione dopo l'orgasmo, {_COERENZA}"""

    # ==================== LESBO ====================
    elif 'pussy_licking_klein_v1' in selected_lora:
        prompt += f"""pussy_licking_klein_v1, scena lesbo intima, ragazza (image 2) sdraiata con gambe divaricate mentre riceve cunnilingus, 
vulva completamente esposta e bagnata, viso dell'altra ragazza affondato tra le cosce, entrambe nude, {location_desc}, {_COERENZA}"""

    # ==================== POSE ====================
    elif 'F2K_Pose_Presenting' in selected_lora or 'Presenting_ass' in selected_lora:
        prompt += f"""F.2K_Pose_Presenting_ass_and_pussy_sideways_epoch_32, ragazza del image 2 sdraiata su un fianco completamente nuda, 
mani dietro le gambe, sedere e vulva in mostra verso lo spettatore, {location_desc}, 
fessura vaginale visibile, posa provocante, {_COERENZA}"""

    # ==================== PENETRATION ====================
    elif 'FK_missionary' in selected_lora:
        prompt += f"""FK_missionary, posizione del missionario, rapporto vaginale in prima persona, ragazza del image 2 totalmente nuda sdraiata sulla schiena con gambe divaricate su {luogo_azione}, 
{location_desc}, seno nudo naturale, capezzoli visibili, penetrazione vaginale profonda, sguardo verso la telecamera, {_COERENZA}"""

    elif 'pov_squatting_sex' in selected_lora:
        prompt += f"""pov_squatting_sex_f2k9b_000002500, inquadratura dal basso, ragazza del image 2 accovacciata sul busto di un uomo sopra {luogo_azione}, pene eretto che penetra la vagina, 
lei guarda verso la telecamera con espressione scioccata e bocca aperta, {location_desc}, 
totalmente nuda, seno naturale, capezzoli visibili, dettagli genitali femminili, {_COERENZA}"""

    elif 'klein-fnelson' in selected_lora or 'fullnelson' in selected_lora.lower():
        prompt += f"""klein-fnelson-13epoc-k3nk, FU11N31S0N, ragazza del image 2 in posizione full nelson su {luogo_azione}, gambe spalancate, penetrazione vaginale profonda, 
{location_desc}, espressioni di piacere intenso, corpo sudato, alta intensità, {_COERENZA}"""

    elif 'full_nelson_anal' in selected_lora:
        prompt += f"""full_nelson_anal_v1_f2k9b_000003100, posizione full nelson anale, ragazza del image 2 su {luogo_azione}, braccia sotto le ginocchia, mani dietro il collo, 
penetrazione anale profonda, {location_desc}, lei si stringe i glutei, espressione di godimento, totalmente nuda, {_COERENZA}"""

    elif 'FK_bulldoganalsex' in selected_lora:
        prompt += f"""FK_bulldoganalsexfinal, ragazza del image 2 a quattro zampe su {luogo_azione}, penetrata analmente da dietro da un uomo, 
{location_desc}, ano e vagina esposti, penetrazione profonda con pene grosso, espressione di piacere, {_COERENZA}"""

    elif 'FK_analonside' in selected_lora:
        prompt += f"""FK_analonside, ragazza del image 2 sdraiata su un fianco con gambe chiuse su {luogo_azione}, totalmente nuda, 
{location_desc}, uomo che la penetra nell'ano con pene gigantesco, focus sulla penetrazione anale, seno naturale visibile, {_COERENZA}"""

    else:
        prompt += f"""ragazza del image 2, giovane donna neutra, completamente nuda, posa esplicita su {luogo_azione}, 
{location_desc}, dettagli anatomici realistici, photorealistic, raw photo 8k, {_COERENZA}"""

    # ====================== QUALITÀ FINALE ======================
    prompt += ", photorealistic, raw photo, 8k, extremely detailed, cinematic lighting, sharp focus, best quality"

    testo.delete('1.0', tk.END)
    testo.insert('1.0', prompt.strip())


# Crea il Combobox
lora = ttk.Combobox(button_game, values=[])
lora.grid(row=6, column=0, pady=5, padx=5)

# ✅ CHIAMA LA FUNZIONE quando clicchi
lora.bind('<Button-1>', lambda e: load_lora())
lora.bind('<<ComboboxSelected>>', select_lora)
# Opzionale: carica i valori anche al startup
load_lora()
lora.current(0)

char_frame = tk.Frame(button_game, bg='gray')
char_frame.grid(row=7, column=0, pady=(40, 0))

# manteniamo un riferimento alle immagini per evitare che il garbage collector le cancelli
char_images = {}




def toggle_select(frame_canvas, index):
    global selected_char, select_face_var, select_body_var,char_path,lab_character 
    
    # Deseleziona tutti
    for i, canvas in enumerate(char_canvases):
        canvas.master.config(bg='blue')
    
    # Seleziona quello cliccato
    frame_canvas.config(bg='green')
    selected_char = index  # assegna l'indice
    
    # Costruisci il path basato sulla selezione
    if select_body_var.get():
        char_path = f"./character/ch{index + 1}full.png"
    elif select_face_var.get():
        char_path = f"./character/ch{index + 1}face.png"
    else:
        char_path = f"./character/ch{index + 1}full.png"
    
    print(f"character selezionato: {char_path}")
    lab_character.config(text=f"Character: {os.path.basename(char_path).split('.')[0]}")
    lab_character.update_idletasks()


import threading

CHAR_W, CHAR_H = 576, 1024  # risoluzione verticale per foto intera (multipli di 16)

def genera_full_in_thread(prompt, dest_path, full_path, canvas, i):
    """Esegue flux2 in un thread separato, poi aggiorna la UI in modo thread-safe."""
    try:
        flux2(
            prompt,
            steps=DEFAULT_STEPS,
            path_image1=dest_path,
            path_lora=path_lora,
            wc=CHAR_W,
            hc=CHAR_H,
            out_dir=CHAR_DIR,
            name=f'ch{i+1}full'
        )
    finally:
        # rimozione del file temporaneo, sempre eseguita anche in caso di errore in flux2
        if os.path.exists(dest_path):
            os.remove(dest_path)
        # torna sul thread principale di tkinter per qualsiasi aggiornamento futuro della UI
        window.after(0, lambda: print(f"Generazione full character {i+1} completata"))


def drag_drop(event, canvas, i):
    global CHAR_W, CHAR_H
    filepath = event.data.strip('{}')

    if not os.path.isfile(filepath):
        return

    ext = os.path.splitext(filepath)[1]
    dest_path = os.path.join(CHAR_DIR, f"tmp{i}{ext}")
    shutil.copy(filepath, dest_path)

    img_originale = Image.open(dest_path)
    w, h = img_originale.size

    vuole_full = full_var.get()
    vuole_crop = face_var.get()  # checkbox "crop_face" -> forza SOLO il crop, niente generazione flux

    is_quadrata = (w == h)
    is_gia_full = (h > w)  # più alta che larga -> già considerata foto intera

    face_path = os.path.join(CHAR_DIR, f"ch{i+1}face.png")
    full_path = os.path.join(CHAR_DIR, f"ch{i+1}full.png")

    # --- CASO A: immagine più alta che larga -> già una foto intera, nessuna generazione flux ---
    if is_gia_full:
        img_originale.save(full_path)
        img_preview = img_originale.crop((0, 0, w, w))
        img_preview.save(face_path)
        img_originale.close()
        os.remove(dest_path)

    # --- CASO B: checkbox "crop_face" attivo -> SOLO crop, mai generazione flux ---
    elif vuole_crop:
        crop_h = min(w, h)
        img_preview = img_originale.crop((0, 0, w, crop_h))
        img_preview.save(face_path)
        img_originale.close()
        os.remove(dest_path)

    # --- CASO C: quadrata o checkbox "full" richiesto -> genera la full con flux IN UN THREAD SEPARATO ---
    elif is_quadrata or vuole_full:
        img_originale.save(face_path)
        img_preview = img_originale

        thumb = img_preview.resize((dc, dc), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=photo)
        char_images[i] = photo
        canvas.update_idletasks()

        prompt = """Crea un'immagine fotorealistica del soggetto a figura intera, dalla testa ai piedi, in un'unica inquadratura verticale.
        Mantieni massima coerenza con il viso originale: stessi lineamenti, stessi occhi, stessa espressione, stessa capigliatura (colore, lunghezza, texture).
        Mantieni esattamente lo stesso vestito visibile nella foto originale: stesso modello, stesso taglio, stesso colore, stessa texture e stessi dettagli, senza modificarlo o sostituirlo. Se il vestito è corto, completa la figura con le gambe e i piedi nudi visibili, mantenendo coerenza con il tono della pelle originale.
        Postura: il soggetto è in piedi, eretto, frontale verso la camera, braccia rilassate lungo i fianchi, gambe leggermente divaricate o unite in posa naturale.
        Corpo e proporzioni anatomicamente coerenti e realistiche.
        Mantieni esattamente lo stesso sfondo, la stessa illuminazione e la stessa palette cromatica della foto di riferimento, senza alterare colori, saturazione o contrasto.
        Inquadratura a figura intera, piedi visibili e non tagliati, testa non tagliata, margine di spazio sopra la testa e sotto i piedi."""

        img_originale.close()

        # avvia flux2 in un thread separato per non bloccare la UI
        t = threading.Thread(
            target=genera_full_in_thread,
            args=(prompt, dest_path, full_path, canvas, i),
            daemon=True
        )
        t.start()
        return  # anteprima già mostrata sopra, esce qui (dest_path verrà rimosso dal thread a fine generazione)

    # --- CASO D (fallback): più larga che alta, nessun checkbox attivo -> salva originale come face ---
    else:
        img_originale.save(face_path)
        img_preview = img_originale
        img_originale.close()
        os.remove(dest_path)

    # anteprima per i casi A, B e D (il caso C ha già mostrato l'anteprima e fatto return sopra)
    thumb = img_preview.resize((dc, dc), Image.LANCZOS)
    photo = ImageTk.PhotoImage(thumb)
    canvas.delete('all')
    canvas.create_image(0, 0, anchor='nw', image=photo)
    char_images[i] = photo

char_canvases = []
# Nel loop dove crei le canvas:
for i in range(4):
    frame_canvas = tk.Canvas(char_frame, width=dc + border, height=dc + border, bg='blue', highlightthickness=0)
    frame_canvas.grid(row=i, column=0, pady=2)

    char = tk.Canvas(frame_canvas, width=dc, height=dc, bg='red', highlightthickness=0)
    char.place(x=border // 2, y=border // 2)

    # MODIFICA QUI: passa anche l'indice i
    char.bind('<Button-1>', lambda event, fc=frame_canvas, idx=i: toggle_select(fc, idx))
    
    char_canvases.append(char)

char1, char2, char3, char4 = char_canvases

 


def load_characters():
    """Carica all'avvio le immagini face già esistenti su disco, se presenti, in ciascuna canvas."""
    for i, canvas in enumerate(char_canvases):
        face_path = os.path.join(CHAR_DIR, f"ch{i+1}face.png")
        if os.path.exists(face_path):
            img = Image.open(face_path)
            img_thumb = img.resize((dc, dc), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_thumb)

            canvas.delete('all')  # rimuove il testo placeholder "foto character N"
            canvas.create_image(0, 0, anchor='nw', image=photo)
            char_images[i] = photo  # mantiene il riferimento, altrimenti il garbage collector la elimina
            img.close()


load_characters()

frame_blood = tk.Frame(window, background='gray')
frame_blood.grid(row=0, column=3, sticky='n')

blood = tk.Canvas(frame_blood, bg='blue', width=80, height=280)
blood.grid(row=0, column=0, sticky='n', pady=10)

# Verifica che la cartella "ampolla" esista e sia accessibile
ampolla_dir = "ampolla"
if os.path.isdir(ampolla_dir):
    imgbs = [os.path.join(ampolla_dir, img) for img in os.listdir(ampolla_dir)
             if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if imgbs:
        try:
            imgb = Image.open(imgbs[0]).resize((81, 281), Image.BICUBIC)
            img_blood_photo = ImageTk.PhotoImage(imgb)
            blood.create_image(0, 0, anchor='nw', image=img_blood_photo)
            # Previene la garbage collection dell'immagine
            blood.img_blood_photo = img_blood_photo
            imgb.close()
        except Exception as e:
            print(f"Errore caricando immagine ampolla: {e}")
else:
    print('Cartella "ampolla" non trovata.')

# Correzione sintassi e gestione immagini ampollach in modo analogo ad "ampolla"

ampollach_dir = "ampollach"
if os.path.isdir(ampollach_dir):
    imgbsch = [os.path.join(ampollach_dir, img) for img in os.listdir(ampollach_dir)
               if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if imgbsch:
        try:
            imgch = Image.open(imgbsch[0]).resize((80, 30), Image.BICUBIC)
            img_ch_photo = ImageTk.PhotoImage(imgch)

            blood_ch1 = tk.Canvas(frame_blood, bg='pink', width=80, height=30)
            blood_ch1.grid(row=1, column=0, sticky='n', pady=20)
            blood_ch1.create_image(0, 0, anchor='nw', image=img_ch_photo)
            blood_ch1.img_ch_photo = img_ch_photo  # Evita garbage collection
            blood_ch1.update_idletasks()

            # Per gli altri 3 canvases (tutti usano la stessa immagine)
            blood_ch2 = tk.Canvas(frame_blood, bg='pink', width=80, height=30)
            blood_ch2.grid(row=2, column=0, sticky='n', pady=70)
            blood_ch2.create_image(0, 0, anchor='nw', image=img_ch_photo)
            blood_ch2.img_ch_photo = img_ch_photo
            blood_ch2.update_idletasks()

            blood_ch3 = tk.Canvas(frame_blood, bg='pink', width=80, height=30)
            blood_ch3.grid(row=3, column=0, sticky='n', pady=50)
            blood_ch3.create_image(0, 0, anchor='nw', image=img_ch_photo)
            blood_ch3.img_ch_photo = img_ch_photo
            blood_ch3.update_idletasks()

            blood_ch4 = tk.Canvas(frame_blood, bg='pink', width=80, height=30)
            blood_ch4.grid(row=4, column=0, sticky='n', pady=50)
            blood_ch4.create_image(0, 0, anchor='nw', image=img_ch_photo)
            blood_ch4.img_ch_photo = img_ch_photo
            blood_ch4.update_idletasks()

            imgch.close()
        except Exception as e:
            print(f"Errore caricando immagine ampollach: {e}")
    else:
        print('Nessuna immagine trovata in "ampollach".')
else:
    print('Cartella "ampollach" non trovata.')

window.mainloop()
