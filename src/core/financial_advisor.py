"""
Financial Advisor Agent Module.
Versione modificata per utilizzare il nuovo RAGService basato su LangChain
in modo trasparente, mantenendo la stessa interfaccia esterna.
"""
import json
import logging
from typing import List, Optional

from datapizza.agents import Agent
from datapizza.memory import Memory

from ..clients import get_client
from ..models import FinancialProfile, PACMetrics, Portfolio
from ..retrieval.rag_service import RAGService
from ..tools import analyze_financial_asset
from .base_agent import BaseAgent

######################## TODO: traduci commenti dall'italiano all'inglese #################

logger = logging.getLogger(__name__)


class FinancialAdvisorAgent(BaseAgent):
    """
    Agente specializzato nella generazione di portafogli.
    Ora utilizza internamente il RAGService potenziato.
    """

    def __init__(
        self,
        name: Optional[str] = "FinancialAdvisorAgent",
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        planning_interval: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        # --- INIZIALIZZAZIONE ESPLICITA ---
        # Invece di chiamare super() per primo, impostiamo gli attributi di base qui.
        self.name = name or self.__class__.__name__
        self.provider = provider
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.planning_interval = (
            planning_interval or self._get_default_planning_interval()
        )
        self.max_steps = max_steps or self._get_default_max_steps()

        # Carica i prompt specifici
        self.rag_query_builder_prompt = self._load_prompt_template("rag_query_builder")
        self.portfolio_extraction_prompt = self._load_prompt_template(
            "portfolio_extraction"
        )
        self.financial_profile_extraction_prompt = self._load_prompt_template(
            "financial_profile_extraction"
        )
        self.pac_metrics_extraction_prompt = self._load_prompt_template(
            "pac_metrics_extraction"
        )

        # Crea il client
        self._client = get_client(provider=provider)

        # Crea la memoria
        self.memory = Memory()

        # Definisci i tools
        financial_tools = [analyze_financial_asset]

        # Crea l'agente datapizza con tutti i pezzi
        self.agent = Agent(
            name=self.name,
            client=self._client,
            memory=self.memory,
            system_prompt=self.system_prompt,
            tools=financial_tools,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
        )

        # Inizializza il RAG service interno
        try:
            self._rag_service = RAGService()
        except Exception as e:
            logger.error(
                f"FALLIMENTO CRITICO: Impossibile inizializzare il RAGService interno: {e}",
                exc_info=True,
            )
            self._rag_service = None

    def _get_default_tools(self) -> list:
        # return [analyze_financial_asset]
        return []

    def _get_default_planning_interval(self) -> int:
        return 1

    def _get_default_max_steps(self) -> int:
        return 3

    def _get_default_system_prompt(self) -> str:
        return self._load_system_prompt()

    # =============== TODO: controlla funzione + traduci in inglese ===============#
    def generate_balanced_portfolio(self, financial_profile: dict) -> dict:
        """
        Genera un portafoglio d'investimento bilanciato basato sul profilo finanziario.
        """
        if not financial_profile:
            logger.error("Il profilo finanziario non può essere None o vuoto")
            raise ValueError("Il profilo finanziario non può essere None o vuoto")

        logger.info(">>> FASE 1: Inizio generazione portafoglio con contesto RAG")

        try:
            profile_json = json.dumps(financial_profile, indent=2)
            logger.debug("--- PROFILO UTENTE (JSON) ---\n%s", profile_json)

            asset_context = ""
            if self._rag_service:
                logger.info(">>> FASE 2: Interrogazione del RAG Service")

                query = self.rag_query_builder_prompt.format(
                    financial_profile=profile_json,
                )

                logger.debug("--- QUERY GENERATA PER IL RAG ---\n%s", query)

                try:
                    retrieved_assets = self._rag_service.retrieve_context(query, k=15)

                    logger.debug(
                        "--- DOCUMENTI RECUPERATI DAL RAG (%d) ---",
                        len(retrieved_assets),
                    )
                    for i, asset in enumerate(
                        retrieved_assets[:3]
                    ):  # Stampa solo i primi 3 per brevità
                        logger.debug(
                            f"  > DOC {i+1} | ID: {asset.get('id')} | Score: {asset.get('score'):.4f}\n    Text: {asset.get('text', '')[:100]}..."
                        )

                    asset_texts = []
                    for asset in retrieved_assets:
                        asset_texts.append(
                            f"[Asset: {asset.get('id')} (relevance: {asset.get('score', 0):.2f})]\n{asset.get('text', '')[:300]}..."
                        )
                    asset_context = "\n---\n".join(asset_texts)
                    logger.info(
                        "Contesto RAG preparato, lunghezza: %d caratteri",
                        len(asset_context),
                    )

                except Exception as e:
                    logger.warning("Recupero RAG fallito: %s", str(e))
            else:
                logger.warning(
                    "RAG retriever non disponibile, procederò senza contesto aggiuntivo."
                )

            extraction_prompt = self.portfolio_extraction_prompt.format(
                client_profile=profile_json, asset_context=asset_context
            )

            logger.info(
                ">>> FASE 3: Invio richiesta all'LLM per la generazione del portafoglio"
            )
            logger.debug(
                "--- PROMPT FINALE PER L'LLM (primi 500 caratteri) ---\n%s...",
                extraction_prompt[:500],
            )

            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=Portfolio,
            )

            if hasattr(response, "structured_data") and response.structured_data:
                portfolio = response.structured_data[0]
                logger.info(">>> FASE 4: Portafoglio generato con successo!")
                logger.info(
                    "Livello di rischio del portafoglio: %s", portfolio.risk_level
                )
                return portfolio.model_dump(mode="json")
            else:
                logger.error(
                    "Generazione portafoglio fallita: nessun dato strutturato ricevuto dall'LLM."
                )
                raise ValueError(
                    "Nessun dato strutturato ricevuto per la generazione del portafoglio"
                )

        except Exception as e:
            logger.error(
                "FALLIMENTO CRITICO in generate_balanced_portfolio: %s",
                str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Impossibile generare il portafoglio: {e}") from e

    # ==================== PAC Metrics Extraction ====================

    def extract_pac_metrics(self, financial_profile: dict) -> PACMetrics:
        """
        Extract PAC (Piano di Accumulo del Capitale) metrics from financial profile.

        Uses structured response to get:
        - Initial investment amount (from savings)
        - Monthly savings capacity (from income - expenses)

        Args:
            financial_profile: Dictionary with financial profile information

        Returns:
            PACMetrics object with initial_investment and monthly_savings

        Raises:
            RuntimeError: If extraction fails
        """
        logger.debug("Extracting PAC metrics from financial profile")

        try:
            extraction_prompt = self.pac_metrics_extraction_prompt.format(
                financial_profile=json.dumps(financial_profile, indent=2)
            )
            logger.debug(
                "Calling structured_response with PACMetrics model with prompt: %s",
                extraction_prompt,
            )

            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=PACMetrics,
                memory=self._memory,
            )

            logger.debug("Structured response received")

            if hasattr(response, "structured_data") and response.structured_data:
                metrics = response.structured_data[0]
                logger.info(
                    "PAC metrics extracted successfully: Initial €%d, Monthly €%.0f",
                    metrics.initial_investment,
                    metrics.monthly_savings,
                )
                return metrics
            else:
                logger.error("No structured data in response")
                logger.warning("Returning default PAC metrics")
                return PACMetrics()

        except Exception as e:
            logger.error("Failed to extract PAC metrics: %s", str(e), exc_info=True)
            logger.warning("Returning default PAC metrics due to extraction failure")
            return PACMetrics()

    def analyze_asset(self, ticker: str, years: int = 10) -> dict:
        """
        Analyze a financial asset and extract structured return data.

        Directly calls the analyze_financial_asset tool and parses the JSON response.
        No need for agent orchestration since the tool does the work directly.

        Args:
            ticker: The stock/ETF ticker symbol (e.g., 'SWDA', 'IX5A')
            years: Number of years to analyze (default: 10)

        Returns:
            dict: Asset analysis data as dictionary with returns, dates, and metrics

        Raises:
            Exception: If the analysis fails
        """
        logger.debug("Analyzing asset %s for %d years", ticker, years)

        try:
            # Call the tool directly - it returns JSON string with all analysis
            logger.debug("Calling analyze_financial_asset tool for %s", ticker)
            result_json = analyze_financial_asset(ticker=ticker, years=years)

            logger.debug("Tool returned JSON, parsing...")

            # Parse the JSON string returned by the tool
            analysis_dict = json.loads(result_json)

            logger.info("Asset analysis completed for %s", ticker)
            logger.debug("Analysis keys: %s", list(analysis_dict.keys()))

            return analysis_dict

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from tool result for %s: %s", ticker, str(e)
            )
            raise RuntimeError(f"Invalid JSON from tool for {ticker}: {e}") from e
        except Exception as e:
            logger.error(
                "Failed to analyze asset %s: %s", ticker, str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to analyze asset {ticker}: {e}") from e
