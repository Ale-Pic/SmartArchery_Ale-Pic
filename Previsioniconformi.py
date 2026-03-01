from ultralytics import YOLO
import cv2
from os import listdir
import math

# Viene chiesto all'utente di scegliere il livello alfa per la conformal prediction.
alfa = float(input("Quale alfa vuoi usare?"))
# Viene caricato il modello.
modello = YOLO("runs/detect/train12/weights/best.pt")
# Creiamo una lista coi percorsi di tutte le immagini del calibration set.
percorsicalibrazione = listdir("Calibrazione/train/images")
# Creiamo parallelamente una lista coi percorsi delle loro etichette.
etichettecalibrazione = [f"{"".join(list(percorso)[:-4])}.txt" for percorso in percorsicalibrazione]
# I livelli lambda della conformal prediction che considereremo sono 0.0, 0.01, 0.02 e così via.
possibililambda = [n/100 for n in range(100)]
# Creiamo una lista che dovrà contenere le bounding box predette da YOLO per ogni immagine.
tutteprevisioni = list()
# Per ogni immagine nel calibration set,
for percorso in percorsicalibrazione:
    # leggiamo l'immagine,
    immagine = cv2.imread(f"Calibrazione/train/images/{percorso}")
    # ne registriamo le dimensioni,
    altezzaimmagine, larghezzaimmagine = immagine.shape[:2]
    # e le prepariamo una lista che dovrà contenere le previsioni.
    scatole = list()
    # In particolare, prendiamo le bounding box di tutte le previsioni fatte dal modello
    # con qualsiasi livello di confidenza (quindi da 0 in su).
    for scatola in modello(immagine, conf=0.0, verbose=False)[0].boxes:
        # Estraiamo le coordinate, espresse in numero di pixel, della bounding box.
        coordinate = scatola.xyxy[0].tolist()
        # Ora le convertiamo, creando una lista che contiene l'ID della classe, le coordinate
        # dell'angolo in alto a sinistra e dell'angolo in basso a destra espresse come frazione
        # della larghezza e dell'altezza dell'immagine, e infine il punteggio di confidenza
        # della bounding box.
        scatole.append([int(scatola.cls), coordinate[0]/larghezzaimmagine, coordinate[1]/altezzaimmagine,
                        coordinate[2]/larghezzaimmagine, coordinate[3]/altezzaimmagine, scatola.conf])
    # Ora che la lista di tutte le previsioni per l'immagine è pronta, la mettiamo nella lista di prima.
    tutteprevisioni.append(scatole)
# Ora leggiamo tutti i file delle etichette: ognuno sarà composto da una serie di righe del tipo
# "classe xcentro ycentro larghezza altezza". Quindi creiamo una lista che contiene un'altra lista
# per ogni immagine, che a sua volta contiene un'altra lista di cinque elementi per ogni riga,
# e dunque per ogni oggetto effettivamente annotato.
tutteetichette = [[riga.split(" ") for riga in open(f"Calibrazione/train/labels/{etichetta}", "r")]
                  for etichetta in etichettecalibrazione]
# Adesso convertiamo la lista in modo che abbia lo stesso formato delle previsioni.
# Per ogni immagine,
for percorso in range(len(tutteetichette)):
    # e per ogni oggetto,
    for lista in range(len(tutteetichette[percorso])):
        # da che erano tutte stringhe poiché stavamo leggendo un file di testo,
        # convertiamo tutti gli elementi in numeri:
        manipolando = [float(elemento) for elemento in tutteetichette[percorso][lista]]
        x_centro, y_centro, w, h = manipolando[1], manipolando[2], manipolando[3], manipolando[4]
        # adesso sostituiamo la lista di stringhe con una lista del tipo [classe, x1, y1, x2, y2].
        tutteetichette[percorso][lista] = [int(manipolando[0]), x_centro - w / 2, y_centro - h / 2,
                                           x_centro + w / 2, y_centro + h / 2]
# Questo lambdaconforme sarà il lambda che useremo per le previsioni. Lo inizializziamo
# col valore più conservativo possibile, ossia 0.99.
lambdaconforme = 0.99
# Ora, per ogni lambda che esaminiamo,
for lambdino in possibililambda:
    # vogliamo registrare il tasso di falsi negativi per ogni immagine:
    tassifalsinegativi = list()
    for percorso in range(len(percorsicalibrazione)):
        # Iniziamo impostando: un contatore per i falsi negativi, un contatore per tutti gli oggetti
        # nell'immagine, un contenitore per le previsioni che sono già state assegnate a un oggetto
        # (in modo che più oggetti non siano abbinati alla stessa previsione), e un contenitore
        # per le previsioni che superano il livello di confidenza 1-lambda.
        falsinegativi = 0
        istanze = 0
        usate = set()
        previsioni = list()
        # Mettiamo in "previsioni" tutte le previsioni per l'immagine la cui confidenza è
        # maggiore di 1-lambda; quando lo facciamo, tralasciamo la confidenza della previsione,
        # così che resti solo una lista [classe, x1, y1, x2, y2].
        for previsione in tutteprevisioni[percorso]:
            if previsione[5] > 1-lambdino:
                previsioni.append(previsione[:-1])
        # Se a tale livello di confidenza non c'è nessuna previsione, vuol dire che nessun oggetto
        # è stato rilevato, quindi il tasso di falsi negativi è 1.
        if len(previsioni) == 0:
            tassifalsinegativi.append(1.0)
            continue
        # Ora esaminiamo ciascun oggetto nell'immagine per vedere se è stato rilevato.
        for oggetto in tutteetichette[percorso]:
            # Aumentiamo di 1 il contatore di oggetti.
            istanze += 1
            # Assumiamo che l'oggetto non sia rilevato fino a prova contraria.
            abbinato = False
            # Scorriamo le previsioni una per una per vedere se una combacia con l'oggetto.
            # Il comando "enumerate" crea una lista del tipo [(0, primaprevisione), (1, secondaprevisione)...]
            for indice, previsione in enumerate(previsioni):
                # Se la previsione è già stata abbinata a un altro oggetto nell'immagine,
                # non la consideriamo.
                if indice in usate:
                    continue
                # Allo stesso modo, consideriamo solo le previsioni della stessa classe
                # dell'oggetto che stiamo esaminando.
                if oggetto[0] != previsione[0]:
                    continue
                # Primo criterio: se la bounding box dell'oggetto è completamente contenuta
                # nella previsione, diamo l'oggetto per rilevato. Aggiungiamo la previsione
                # a quelle usate, marchiamo "abbinato" come vero in modo che non sia contato
                # un falso negativo, e tralasciamo di esaminare le altre previsioni, perché
                # una adatta è già stata trovata.
                if oggetto[3] <= previsione[3] and oggetto[1] >= previsione[1] and oggetto[4] <= previsione[4] and oggetto[2] >= previsione[2]:
                    abbinato = True
                    usate.add(indice)
                    break
                # Secondo criterio: se i centri delle due bounding box sono sufficientemente vicini,
                # ossia se sono separati da un segmento non più lungo di 0.02, l'oggetto viene
                # considerato rilevato. Stesse operazioni di prima.
                centrooggetto = ((oggetto[1]+oggetto[3])/2, (oggetto[2]+oggetto[4])/2)
                centroprevisione = ((previsione[1]+previsione[3])/2, (previsione[2]+previsione[4])/2)
                if ((centrooggetto[0]-centroprevisione[0])**2+(centrooggetto[1]-centroprevisione[1])**2)**0.5 <= 0.02:
                    abbinato = True
                    usate.add(indice)
                    break
                # Terzo criterio (IoU): se la proporzione tra l'area di intersezione e quella di unione
                # delle due bounding box supera 0.3, l'oggetto viene contato come rilevato.
                larghezzaintersezione = max(0, min(oggetto[3], previsione[3]) - max(oggetto[1], previsione[1]))
                altezzaintersezione = max(0, min(oggetto[4], previsione[4]) - max(oggetto[2], previsione[2]))
                areaintersezione = larghezzaintersezione * altezzaintersezione
                areaunione = (oggetto[3] - oggetto[1]) * (oggetto[4] - oggetto[2]) + (previsione[3] - previsione[1]) * (
                            previsione[4] - previsione[2]) - areaintersezione
                if areaintersezione / areaunione > 0.3:
                    abbinato = True
                    usate.add(indice)
                    break
            # Se dopo aver controllato tutte le previsioni l'oggetto ancora non è stato
            # abbinato a nessuna previsione, allora lo consideriamo come un falso negativo.
            if not abbinato:
                falsinegativi += 1
        # A questo punto calcoliamo semplicemente il tasso di falsi negativi dell'immagine
        # e lo salviamo nella lista apposita.
        tassifalsinegativi.append(falsinegativi/istanze)
    # Ora possiamo calcolare la quantità \hat{R}_n(\lambda), che non è altro che la media
    # dei tassi di falsi negativi di tutte le immagini.
    rischio = sum(tassifalsinegativi)/len(tassifalsinegativi)
    # Se la statistica usata nel paper è minore di alfa, allora abbiamo trovato il nostro lambda.
    if len(percorsicalibrazione)/(len(percorsicalibrazione)+1)*rischio+1/(len(percorsicalibrazione)+1) <= alfa:
        lambdaconforme = lambdino
        break
# Ora carichiamo l'immagine su cui fare la previsione.
immagine = cv2.imread("C:\\Users\\aless\\OneDrive\\Desktop\\Gastone\\Buone\\Buonissime\\20241014_204138.jpg")
# Estraiamo le previsioni con confidenza superiore a 1-lambda.
risultati = modello(immagine, conf=1-lambdaconforme, verbose=False)
# Separiamo i bersagli dalle frecce.
bersagli = list()
frecce = list()
for box in risultati[0].boxes:
    if box.cls[0] == 2 or box.cls[0] == 0:
        bersagli.append(box.xyxy.tolist()[0])
    elif box.cls[0] == 1:
        frecce.append(box.xyxy.tolist()[0])
# Ora prendiamo ogni freccia e calcoliamo il centro della bounding box: useremo quel punto
# per calcolare il punteggio.
centrifrecce = list()
for freccia in frecce:
    centrifrecce.append([(freccia[0]+freccia[2])/2, (freccia[1]+freccia[3])/2])
# Ora associamo ciascuna freccia al suo bersaglio, controllando che il centro della prima sia contenuto
# nella bounding box del secondo. Se la freccia non si trova su un bersaglio, la marchiamo come "None",
# altrimenti le assegniamo un numero corrispondente al bersaglio.
bersaglifrecce = list()
for freccia in centrifrecce:
    assegnato = False
    for bersaglio in bersagli:
        if bersaglio[0] <= freccia[0] <= bersaglio[2] and bersaglio[1] <= freccia[1] <= bersaglio[3]:
            bersaglifrecce.append(bersagli.index(bersaglio))
            assegnato = True
            break
    if not assegnato:
        bersaglifrecce.append(None)
# Finalmente possiamo calcolare i punteggi.
punteggifrecce = list()
# Per ogni freccia:
for freccia in centrifrecce:
    # ripeschiamo il bersaglio che le è associato; se non ce n'è nessuno, vuol dire
    # che la freccia è fuori bersaglio e quindi ha fatto 0 punti.
    bersaglio = bersaglifrecce[centrifrecce.index(freccia)]
    if bersaglio is None:
        punteggifrecce.append(0)
        continue
    # Se invece si trova su un bersaglio, calcoliamo il punteggio: poniamo un sistema di coordinate
    # cartesiane in cui il centro del bersaglio è l'origine, e i suoi bordi corrispondono alle rette
    # x=-1, x=1, y=-1, y=1. Si verifica facilmente che allora il punteggio è funzione della norma
    # del punto corrispondente alla freccia: più è lontana dall'origine, minore è il punteggio.
    normalizzataorizzontale = (freccia[0]-bersagli[bersaglio][0])/(bersagli[bersaglio][2]-bersagli[bersaglio][0])
    normalizzataverticale = (freccia[1]-bersagli[bersaglio][1])/(bersagli[bersaglio][3]-bersagli[bersaglio][1])
    norma = ((normalizzataorizzontale-0.5)**2+(normalizzataverticale-0.5)**2)**0.5
    # In particolare, il punteggio si può ottenere tramite questa formula.
    punteggio = 10-math.floor(20*norma)
    # Se la freccia sfora dai cerchi concentrici e quindi la norma è maggiore di 1, il punteggio è 0.
    if punteggio < 0:
        punteggifrecce.append(0)
    else:
        punteggifrecce.append(punteggio)
# Mostriamo i punteggi predetti e le previsioni sull'immagine.
print(punteggifrecce)
risultati[0].show()
# Fine!
