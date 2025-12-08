# Lokalplanens Anatomi Anno 2025
Dette materiale er udarbejdet som en del af projektet '*Lokalplanens Anatomi Anno 2025*' på 3. semester af Surveying, Planning, and Land Management ved Aalborg Universitet af August Sawman Nygaard og Tobias Faurhøj Knudsen.  
Materialet indeholder et tekstkorpus bestående af alle lokalplaner pr. 3/10-2025 samt scripts, der kan bruges til analyser af tekstkorpusset.

I forbindelse med projektet er der desuden udarbejdet interaktive word embedding plots, som kan ses [her](https://tobiasfknudsen.github.io/lokalplankorpus/).

## Mappestruktur og indhold
Materialet indeholder følgende mapper og filer:  

- `text_preprocessing/` indeholder scripts, der kan bruges til forbehandling af lokalplankorpusset, herunder tokenisering, rensning og lemmatisering.  
- `leksikalske_analyser/` indeholder scripts, der kan bruges til leksikalske analyser af lokalplankorpusset, herunder Wordclouds og beregning af ordfrekvens.  
- `semantiske_analyser` indeholder scripts, der kan bruges til semantiske analyser af lokalplankorpusset, herunder træning af Word2Vec-modeller, interaktive word embedding plots og opslag i modellerne.  

Filerne `stopord.txt` indeholder listen over stopord anvendt i tekstforbehandlingen, `index.html` er startsiden til de interaktive visualiseringer via GitHub Pages, og `README.md` er denne fil, som introducerer materialet og forklarer indholdet.

## Database
Filen `lokalplankorpus.db.zip` indeholder lokalplankorpusset anvendt i projektet.  
Efter udpakning findes databasen `lokalplaner.db` med tabellen `lokalplaner`. Tabellen indeholder flere kolonner, herunder følgende med tekst fra lokalplanerne:  

- `tekst` indeholder rå tekst direkte fra OCR  
- `tekst_renset` indeholder renset tekst, for at fjerne støj  
- `tekst_renset_lemma` indeholder tekst renset og efterfølgende lemmatiseret  


