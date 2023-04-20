# Shortest Path Race Route Calculator

Tato aplikace vypočítává nejkratší trasu závodu na základě zadaných bodů startu, cíle a stanovišť. Aplikace používá OpenStreetMap data, DMR5G výšková data a vlastní algoritmy pro výpočet nejkratších cest mezi body.

### Instalace

1. Naklonujte repozitář na svůj počítač.
2. Nainstalujte všechny potřebné knihovny pomocí `pip install -r requirements.txt`.
3. Spusťte aplikaci pomocí příkazu `streamlit run analyza_trasy.py`.

### Použití

1. Načtěte uloženou variantu bodů nebo upravte body v tabulce přímo v aplikaci.
2. Zadejte název nové varianty bodů, pokud jste provedli změny, a uložte jako samostatnou verzi.
3. Vyberte počet vrácených tras a případně nastavte maximální počet tras.
4. Vyberte startovní bod.
5. Posuňte posuvníkem pro výběr trasy podle pořadí.
6. Aplikace zobrazí mapu s vybranou trasou, časovou délku trasy a seznam bodů.

### Funkce kódu

Výše uvedený kód je část aplikace, která zajišťuje interakci s uživatelem. Některé klíčové funkce zahrnují:

- Načítání a úprava bodů z .csv souborů.
- Stahování OSM dat a DMR5G výškových dat.
- Přidání výškových dat ke grafu cest.
- Výpočet nejkratších cest mezi uzly pomocí Floyd-Warshall algoritmu.
- Výpočet délky cesty pro každou kombinaci bodů.
- Výběr nejkratší cesty a zobrazení mapy s vybranou trasou.

### Licence

Tento projekt je šířen pod MIT licencí. Viz soubor LICENSE pro více informací.

**Toto README.md bylo vygenerováno pomocí [chat.openai.com](https://chat.openai.com)**