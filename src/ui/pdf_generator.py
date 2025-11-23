import io
import logging
from datetime import datetime
from typing import Any, Dict

import matplotlib.colors as mcolors

# Import necessari per il grafico
import matplotlib.pyplot as plt
from fpdf import FPDF

logger = logging.getLogger(__name__)


class PDFGenerator(FPDF):
    """Classe personalizzata per la generazione del report finanziario."""

    def __init__(
        self,
        agent_config: Dict[str, str],
        profile_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
    ):
        super().__init__()
        self.agent_config = agent_config
        self.profile_data = profile_data
        self.portfolio_data = portfolio_data
        self.title_text = "Report di Analisi Finanziaria e Portafoglio AI"

    def header(self):
        # Logo o Titolo
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, self.title_text, 0, 1, "C")
        self.ln(5)

    def footer(self):
        # Numero di pagina
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        # Intestazione del capitolo
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, "L", 1)
        self.ln(4)

    def chapter_body(self, text):
        # Corpo del capitolo
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, text)
        self.ln()

    def section_subtitle(self, title):
        # Sottotitolo per le sezioni interne
        self.set_font("Arial", "B", 11)
        self.cell(0, 6, title, 0, 1, "L")
        self.ln(2)

    def _create_pie_chart(self) -> io.BytesIO:
        """
        Genera un grafico a torta usando Matplotlib e restituisce il buffer dell'immagine.
        """
        assets = self.portfolio_data.get("assets", [])

        # Prepara i dati
        labels = [asset.get("symbol", "Asset") for asset in assets]
        sizes = [asset.get("percentage", 0) for asset in assets]

        # Configurazione grafico
        fig, ax = plt.subplots(figsize=(6, 4))

        # Colori professionali
        # Se ci sono pochi asset usa Paired, altrimenti default
        if len(labels) <= 12:
            colors = plt.cm.Paired(range(len(labels)))
        else:
            colors = None

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 9},
        )

        ax.axis("equal")  # Assicura che la torta sia circolare
        plt.title("Allocazione Portafoglio", fontsize=12, pad=20)

        # Salva in memoria
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=100)
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer

    def add_cover_page(self):
        self.add_page()
        self.set_font("Arial", "B", 24)
        self.cell(0, 40, self.title_text, 0, 1, "C")

        self.set_font("Arial", "", 12)
        self.ln(20)

        self.chapter_title("Dettagli Generazione")
        self.chapter_body(f"Data e Ora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        self.chapter_body(f"Agente: {self.agent_config.get('name', 'N/A')}")
        self.chapter_body(
            f"Provider LLM: {self.agent_config.get('provider', 'N/A').upper()}"
        )
        self.chapter_body(f"Modello: {self.agent_config.get('model', 'N/A')}")

        self.ln(20)
        self.chapter_title("Dichiarazione")
        self.chapter_body(
            "Questo documento è una raccomandazione generata dall'AI e non costituisce una consulenza finanziaria personalizzata. "
            "Consultare un professionista prima di prendere decisioni di investimento."
        )
        self.add_page()

    def add_profile_summary(self):
        self.chapter_title("1. Riepilogo del Profilo Finanziario")

        main_info = [
            (
                "Età e Lavoro",
                f"{self.profile_data.get('age_range', 'N/A')}, {self.profile_data.get('employment_status', 'N/A')}",
            ),
            ("Reddito Annuo", self.profile_data.get("annual_income_range", "N/A")),
            ("Obiettivi", self.profile_data.get("goals", "N/A")),
            ("Tolleranza al Rischio", self.profile_data.get("risk_tolerance", "N/A")),
            (
                "Esperienza Investimento",
                self.profile_data.get("investment_experience", "N/A"),
            ),
        ]

        for label, value in main_info:
            self.set_font("Arial", "B", 10)
            self.cell(50, 6, f"{label}:", 0, 0, "L")
            self.set_font("Arial", "", 10)
            self.cell(0, 6, str(value), 0, 1, "L")
        self.ln(5)

    def add_portfolio_recommendation(self):
        self.chapter_title("2. Raccomandazione di Portafoglio")

        # --- Sezione 2.1: Strategia e Rischio ---
        self.section_subtitle("Profilo Strategico")

        # Livello di Rischio
        self.set_font("Arial", "B", 10)
        self.cell(40, 6, "Livello di Rischio:", 0, 0, "L")
        self.set_font("Arial", "", 10)
        self.cell(0, 6, self.portfolio_data.get("risk_level", "N/A").upper(), 0, 1, "L")

        # Ribilanciamento
        rebalancing = self.portfolio_data.get("rebalancing_schedule", "N/A")
        self.set_font("Arial", "B", 10)
        self.cell(40, 6, "Ribilanciamento:", 0, 0, "L")
        self.set_font("Arial", "", 10)
        # Usa multi_cell per sicurezza se il testo è lungo
        current_x = self.get_x()
        current_y = self.get_y()
        self.multi_cell(0, 6, str(rebalancing), 0, "L")

        self.ln(2)

        # Logica Strategica
        self.set_font("Arial", "B", 10)
        self.cell(0, 6, "Logica di Investimento:", 0, 1, "L")
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, self.portfolio_data.get("portfolio_reasoning", "N/A"))
        self.ln(5)

        # --- Sezione 2.2: Considerazioni Chiave ---
        considerations = self.portfolio_data.get("key_considerations", [])
        if considerations:
            self.section_subtitle("Considerazioni Chiave")
            self.set_font("Arial", "", 10)

            for item in considerations:
                # 1. Reset X al margine sinistro
                self.set_x(self.l_margin)

                # 2. Disegna Bullet
                # Il bullet occupa 5mm
                self.cell(5, 5, "-", 0, 0, "R")

                # 3. Calcola larghezza disponibile per il testo
                # Larghezza Totale - Margine Destro - Posizione Attuale X
                text_width = self.w - self.r_margin - self.get_x()

                # 4. Disegna il testo
                # Usa text_width esplicito invece di 0 per evitare errori di spazio
                self.multi_cell(text_width, 5, str(item))

            self.ln(5)

        # --- Sezione 2.3: Grafico Allocazione ---
        self.chapter_title("2.1. Allocazione Asset (Grafico)")

        try:
            # Genera il grafico
            chart_buffer = self._create_pie_chart()

            # Calcola posizione per centrare l'immagine
            # Larghezza pagina (A4) ~210mm. Immagine larga 120mm.
            x_img = (self.w - 120) / 2

            # Inserisci immagine dal buffer
            self.image(chart_buffer, x=x_img, w=120)
            self.ln(5)
        except Exception as e:
            logger.error(f"Errore generazione grafico: {e}")
            self.cell(
                0, 10, "Impossibile generare il grafico dell'allocazione.", 0, 1, "C"
            )

        # --- Sezione 2.4: Dettaglio Asset (Lista) ---
        self.section_subtitle("Dettaglio e Giustificazione Asset")

        self.set_font("Arial", "", 10)
        for asset in self.portfolio_data.get("assets", []):
            symbol = asset.get("symbol", "N/A")
            pct = asset.get("percentage", 0)
            justification = asset.get("justification", "N/A")

            # Intestazione Asset
            self.set_font("Arial", "B", 10)
            self.cell(0, 6, f"{symbol} ({pct}%)", 0, 1, "L")

            # Giustificazione (Indentata)
            self.set_font("Arial", "", 10)
            self.set_x(self.l_margin + 5)  # Indentazione 5mm

            # Calcola larghezza ridotta per l'indentazione
            desc_width = self.w - self.r_margin - (self.l_margin + 5)
            self.multi_cell(desc_width, 5, str(justification))

            self.ln(3)  # Spazio tra gli asset

    def add_disclaimer(self):
        self.chapter_title("3. Note e Disclamer")
        self.set_font("Arial", "I", 9)
        self.multi_cell(
            0,
            5,
            "**AVVERTENZA SUL RISCHIO:** I rendimenti passati non sono indicativi di risultati futuri. "
            "Il valore degli investimenti può diminuire o aumentare. Tutte le raccomandazioni si basano su "
            "modelli di intelligenza artificiale e non tengono conto della situazione finanziaria "
            "personale completa di un individuo. Questo report non è da considerarsi consulenza finanziaria "
            "o legale. Consultare sempre un consulente finanziario certificato prima di investire.",
        )
        self.ln(5)

    def generate(self) -> bytes:
        """Genera il PDF completo e restituisce i byte."""
        self.add_cover_page()
        self.add_profile_summary()
        self.add_portfolio_recommendation()
        self.add_disclaimer()

        # Genera l'output come bytes (dest='S' ritorna bytearray o bytes)
        pdf_output = self.output(dest="S")

        # Conversione esplicita in bytes per sicurezza con Streamlit
        return bytes(pdf_output)
