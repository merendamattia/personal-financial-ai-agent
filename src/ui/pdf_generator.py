import logging
from datetime import datetime
from typing import Any, Dict

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
        self.add_page()  # Passa alla pagina successiva

    def add_profile_summary(self):
        self.chapter_title("1. Riepilogo del Profilo Finanziario")

        # Estrai le informazioni principali dal profilo (Flat Map)
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

        # Rischio e Motivazione (Invariato)
        self.set_font("Arial", "B", 10)
        self.cell(50, 6, "Livello di Rischio:", 0, 0, "L")
        self.set_font("Arial", "", 10)
        self.cell(0, 6, self.portfolio_data.get("risk_level", "N/A").upper(), 0, 1, "L")

        self.set_font("Arial", "B", 10)
        self.cell(0, 6, "Logica Strategica:", 0, 1, "L")
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, self.portfolio_data.get("portfolio_reasoning", "N/A"))
        self.ln(5)

        # Tabella di Allocazione
        self.chapter_title("   2.1. Allocazione Asset")

        # Larghezze delle colonne (Larghezze maggiorate)
        W_SYMBOL = 40
        W_PCT = 25
        W_JUST = self.w - self.l_margin - self.r_margin - W_SYMBOL - W_PCT

        # Intestazione tabella
        self.set_fill_color(230, 230, 230)
        self.set_font("Arial", "B", 10)
        self.cell(W_SYMBOL, 10, "Asset", 1, 0, "C", 1)
        self.cell(W_PCT, 10, "Percentuale", 1, 0, "C", 1)
        self.cell(W_JUST, 10, "Giustificazione", 1, 1, "C", 1)

        # Righe della tabella
        self.set_font("Arial", "", 9)
        line_height_mm = self.font_size / self.k * 1.7
        PADDING = 1.5

        for asset in self.portfolio_data.get("assets", []):
            pct = f"{asset.get('percentage', 0):.2f}%"
            symbol = asset.get("symbol", "N/A")
            justification = asset.get("justification", "N/A").replace("\n", " ")

            x_start = self.get_x()
            y_start = self.get_y()

            # --- 1. Disegna Giustificazione per Calcolare l'Altezza Reale (Misurazione) ---

            # Posiziona X all'inizio della terza colonna con padding
            self.set_xy(x_start + W_SYMBOL + W_PCT + PADDING, y_start + PADDING)

            # Disegna testo multi-linea (border=0)
            self.multi_cell(
                W_JUST - (2 * PADDING), line_height_mm, justification, 0, "L", 0
            )

            # Cattura la Y finale impostata da multi_cell
            y_end = self.get_y()

            # Calcola l'altezza della riga
            row_height = max(10.0, y_end - y_start)

            # --- CORREZIONE CHIAVE: Reset Y per eliminare la riga vuota extra ---
            # multi_cell in FPDF aggiunge un piccolo spazio verticale che causa la riga vuota extra.
            # Lo correggiamo forzando y_end a non superare l'altezza calcolata.
            # Se y_end è molto vicino a y_start + row_height, lo forziamo.
            if (
                y_end > y_start + row_height + 0.5
            ):  # 0.5 è un piccolo margine di tolleranza
                # Se l'altezza occupata è troppo grande, la riduciamo all'altezza calcolata.
                # Questo previene salti inutili e righe vuote.
                y_end = y_start + row_height

            # --- 2. Disegno del Contenuto Rimanente (Centratura e Testo Semplice) ---

            v_offset = (row_height - line_height_mm) / 2

            # Colonna 1: Simbolo
            self.set_xy(x_start + PADDING, y_start + v_offset)
            self.cell(W_SYMBOL - (2 * PADDING), line_height_mm, symbol, 0, 0, "L")

            # Colonna 2: Percentuale
            self.set_xy(x_start + W_SYMBOL, y_start + v_offset)
            self.cell(W_PCT, line_height_mm, pct, 0, 0, "C")

            # --- 3. Disegno dei Bordi (con altezza misurata) ---

            self.set_xy(x_start, y_start)

            # Bordo completo della riga (contorno)
            self.rect(x_start, y_start, W_SYMBOL + W_PCT + W_JUST, row_height)

            # Linee verticali divisorie
            self.line(
                x_start + W_SYMBOL, y_start, x_start + W_SYMBOL, y_start + row_height
            )
            self.line(
                x_start + W_SYMBOL + W_PCT,
                y_start,
                x_start + W_SYMBOL + W_PCT,
                y_start + row_height,
            )

            # --- 4. Avanzamento Cursore ---

            # Sposta il cursore X al margine sinistro e Y alla posizione misurata (y_end)
            self.set_xy(self.l_margin, y_end)

        # --- Avanzamento finale del cursore dopo la tabella ---
        self.ln(5)

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

        # Genera l'output come bytearray/bytes
        pdf_output = self.output(dest="S")

        # Conversione esplicita in bytes per Streamlit
        return bytes(pdf_output)
