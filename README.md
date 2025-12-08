# Lokalplanens Anatomi Anno 2025

Dette materiale er udarbejdet som en del af projektet '*Lokalplanens Anatomi Anno 2025*' på 3. semester af Surveying, Planning, and Land Management ved Aalborg Universitet af August Sawman Nygaard og Tobias Faurhøj Knudsen.  
Materialet indeholder et tekstkorpus bestående af alle lokalplaner pr. 3/10-2025 samt scripts, der kan bruges til analyser af tekstkorpusset.

I forbindelse med projektet er der desuden udarbejdet interaktive word embedding plots, som kan ses [her](https://tobiasfknudsen.github.io/lokalplankorpus/).

## Mappestruktur og indhold

- [`text_preprocessing/`](text_preprocessing/)  
  Indeholder scripts, der kan bruges til forbehandling af tekst, fx tokenisering, rensning og lemmatisering.  

- [`leksikalske_analyser/`](leksikalske_analyser/)  
  Indeholder scripts, der kan bruges til leksikalske analyser, fx Wordclouds og beregning af ordfrekvens.  

- [`semantiske_analyser/`](semantiske_analyser/)  
  Indeholder scripts, der kan bruges til semantiske analyser, fx Word2Vec-modeller, interaktive word embedding plots og opslag i modellerne.  

- `stopord.txt` - liste over stopord anvendt i tekstforbehandlingen.  
- `index.html` - startside til de interaktive visualiseringer via GitHub Pages.  
- `README.md` - denne fil, som introducerer repositoryet og forklarer indholdet.  
