from __future__ import annotations

logger = __import__("logging").getLogger(__name__)


def get_eurostoxx50_tickers() -> list[str]:
    """Return European large-cap universe as yfinance ticker symbols (~150 aktier).

    Dækker DAX 40, CAC 40, Euro Stoxx 50, FTSE 100 (top 50), OMX Nordic og SMI.
    Suffixes: .AS=Amsterdam, .PA=Paris, .DE=XETRA, .MI=Milan, .MC=Madrid,
              .SW=Zurich, .CO=Copenhagen, .ST=Stockholm, .OL=Oslo, .HE=Helsinki,
              .L=London
    """
    return list(_EUROPEAN_TICKERS)


_EUROPEAN_TICKERS: list[str] = [
    # -----------------------------------------------------------------------
    # Frankrig — CAC 40
    # -----------------------------------------------------------------------
    "MC.PA",  # LVMH
    "OR.PA",  # L'Oréal
    "TTE.PA",  # TotalEnergies
    "SAN.PA",  # Sanofi
    "AIR.PA",  # Airbus
    "BNP.PA",  # BNP Paribas
    "CS.PA",  # AXA
    "DG.PA",  # Vinci
    "AI.PA",  # Air Liquide
    "SU.PA",  # Schneider Electric
    "CAP.PA",  # Capgemini
    "KER.PA",  # Kering
    "RI.PA",  # Pernod Ricard
    "SGO.PA",  # Saint-Gobain
    "VIE.PA",  # Veolia
    "RMS.PA",  # Hermès
    "DSY.PA",  # Dassault Systèmes
    "EDF.PA",  # EDF (energi)
    "GLE.PA",  # Société Générale
    "ACA.PA",  # Crédit Agricole
    "LR.PA",  # Legrand
    "ML.PA",  # Michelin
    "RNO.PA",  # Renault
    "PUB.PA",  # Publicis
    "VIV.PA",  # Vivendi
    "EN.PA",  # Bouygues
    "SW.PA",  # Sodexo
    "ORA.PA",  # Orange
    "STM.PA",  # STMicroelectronics (også .MI)
    "HO.PA",  # Thales
    # -----------------------------------------------------------------------
    # Tyskland — DAX 40
    # -----------------------------------------------------------------------
    "SAP.DE",
    "SIE.DE",  # Siemens
    "ALV.DE",  # Allianz
    "MBG.DE",  # Mercedes-Benz
    "BMW.DE",
    "BAYN.DE",  # Bayer
    "EOAN.DE",  # E.ON
    "DTE.DE",  # Deutsche Telekom
    "MUV2.DE",  # Munich Re
    "IFX.DE",  # Infineon
    "DB1.DE",  # Deutsche Börse
    "HFG.DE",  # Heidelberg Materials
    "VOW3.DE",  # Volkswagen
    "BAS.DE",  # BASF
    "RWE.DE",  # RWE
    "MRK.DE",  # Merck KGaA
    "DHER.DE",  # Delivery Hero
    "HEN3.DE",  # Henkel
    "ZAL.DE",  # Zalando
    "QGEN.DE",  # Qiagen
    "MTX.DE",  # MTU Aero Engines
    "VNA.DE",  # Vonovia
    "CON.DE",  # Continental
    "AIR.DE",  # Airbus (XETRA)
    "DPW.DE",  # Deutsche Post / DHL
    "DBK.DE",  # Deutsche Bank
    "HEI.DE",  # Heidelberg Cement
    "SHL.DE",  # Siemens Healthineers
    "ENR.DE",  # Siemens Energy
    "P911.DE",  # Porsche AG
    # -----------------------------------------------------------------------
    # Holland
    # -----------------------------------------------------------------------
    "ASML.AS",
    "SHELL.AS",
    "INGA.AS",  # ING
    "PHIA.AS",  # Philips
    "AD.AS",  # Ahold Delhaize
    "RAND.AS",  # Randstad
    "NN.AS",  # NN Group
    "HEIA.AS",  # Heineken
    "WKL.AS",  # Wolters Kluwer
    "AKZA.AS",  # AkzoNobel
    # -----------------------------------------------------------------------
    # Storbritannien — FTSE 100 (top 50)
    # -----------------------------------------------------------------------
    "SHEL.L",  # Shell
    "AZN.L",  # AstraZeneca
    "HSBA.L",  # HSBC
    "ULVR.L",  # Unilever
    "BP.L",  # BP
    "GSK.L",  # GSK
    "RIO.L",  # Rio Tinto
    "BHP.L",  # BHP
    "LLOY.L",  # Lloyds
    "VOD.L",  # Vodafone
    "BARC.L",  # Barclays
    "DGE.L",  # Diageo
    "REL.L",  # RELX
    "NG.L",  # National Grid
    "PRU.L",  # Prudential
    "CRH.L",  # CRH
    "AAL.L",  # Anglo American
    "STAN.L",  # Standard Chartered
    "IMB.L",  # Imperial Brands
    "BATS.L",  # British American Tobacco
    "WPP.L",  # WPP
    "BT-A.L",  # BT Group
    "FLTR.L",  # Flutter Entertainment
    "EXPN.L",  # Experian
    "III.L",  # 3i Group
    "RKT.L",  # Reckitt Benckiser
    "TSCO.L",  # Tesco
    "SKG.L",  # Smurfit Kappa
    "CPG.L",  # Compass Group
    "LSEG.L",  # London Stock Exchange Group
    # -----------------------------------------------------------------------
    # Italien
    # -----------------------------------------------------------------------
    "ENEL.MI",
    "ENI.MI",
    "ISP.MI",  # Intesa Sanpaolo
    "UCG.MI",  # UniCredit
    "STM.MI",  # STMicroelectronics
    "G.MI",  # Generali
    "TIT.MI",  # Telecom Italia
    "LDO.MI",  # Leonardo
    "RACE.MI",  # Ferrari
    "MONC.MI",  # Moncler
    # -----------------------------------------------------------------------
    # Spanien
    # -----------------------------------------------------------------------
    "IBE.MC",  # Iberdrola
    "ITX.MC",  # Inditex
    "SAN.MC",  # Banco Santander
    "BBVA.MC",
    "REP.MC",  # Repsol
    "TEF.MC",  # Telefónica
    "CLNX.MC",  # Cellnex Telecom
    "FER.MC",  # Ferrovial
    # -----------------------------------------------------------------------
    # Schweiz — SMI
    # -----------------------------------------------------------------------
    "NESN.SW",  # Nestlé
    "ROG.SW",  # Roche
    "NOVN.SW",  # Novartis
    "ABBN.SW",  # ABB
    "ZURN.SW",  # Zurich Insurance
    "UBSG.SW",  # UBS
    "CSGN.SW",  # Credit Suisse (afviklet — kan give fejl)
    "GIVN.SW",  # Givaudan
    "SIKA.SW",  # Sika
    "LONN.SW",  # Lonza
    "PGHN.SW",  # Partners Group
    "SLHN.SW",  # Swiss Life
    # -----------------------------------------------------------------------
    # Danmark — OMX Copenhagen
    # -----------------------------------------------------------------------
    "NOVO-B.CO",  # Novo Nordisk
    "MAERSK-B.CO",  # A.P. Møller-Mærsk
    "CARL-B.CO",  # Carlsberg
    "COLO-B.CO",  # Coloplast
    "DEMANT.CO",  # Demant
    "DSV.CO",  # DSV
    "GN.CO",  # GN Store Nord
    "ORSTED.CO",  # Ørsted
    "PNDORA.CO",  # Pandora
    "RBREW.CO",  # Royal Unibrew
    "TRYG.CO",  # Tryg
    "VWS.CO",  # Vestas Wind Systems
    "AMBU-B.CO",  # Ambu
    "CHR.CO",  # Chr. Hansen (nu FMC Chemicals)
    "NETC.CO",  # Netcompany
    # -----------------------------------------------------------------------
    # Sverige — OMX Stockholm
    # -----------------------------------------------------------------------
    "VOLV-B.ST",  # Volvo
    "ERIC-B.ST",  # Ericsson
    "SEB-A.ST",  # SEB
    "SWED-A.ST",  # Swedbank
    "SHB-A.ST",  # Handelsbanken
    "ATCO-A.ST",  # Atlas Copco
    "SKF-B.ST",  # SKF
    "INVE-B.ST",  # Investor AB
    "SAND.ST",  # Sandvik
    "ALFA.ST",  # Alfa Laval
    "HEXA-B.ST",  # Hexagon
    "ESSITY-B.ST",  # Essity
    "TEL2-B.ST",  # Tele2
    "TELIA.ST",  # Telia
    "NDA-SE.ST",  # Nordea (SE)
    "BOL.ST",  # Boliden
    # -----------------------------------------------------------------------
    # Norge — Oslo Børs
    # -----------------------------------------------------------------------
    "EQNR.OL",  # Equinor
    "DNB.OL",  # DNB Bank
    "TEL.OL",  # Telenor
    "ORK.OL",  # Orkla
    "MOWI.OL",  # Mowi (laks)
    "SALM.OL",  # SalMar
    "AKRBP.OL",  # Aker BP
    "YAR.OL",  # Yara International
    "NHY.OL",  # Norsk Hydro
    "SUBC.OL",  # Subsea 7
    # -----------------------------------------------------------------------
    # Finland — OMX Helsinki
    # -----------------------------------------------------------------------
    "NOKIA.HE",
    "STERV.HE",  # Stora Enso
    "UPM.HE",  # UPM-Kymmene
    "KNEBV.HE",  # Kone
    "WRT1V.HE",  # Wärtsilä
    "NESTE.HE",  # Neste (bæredygtig energi)
    "METSO.HE",  # Metso
]


# ---------------------------------------------------------------------------
# Euro Stoxx 50 constituents — yfinance ticker symbols, as of mid-2025.
# Suffixes: .AS=Amsterdam, .PA=Paris, .DE=Frankfurt/XETRA, .MI=Milan,
#           .MC=Madrid, .SW=Zurich, .CO=Copenhagen
# ---------------------------------------------------------------------------
_EUROSTOXX50_TICKERS: list[str] = [
    # Netherlands
    "ASML.AS",
    "SHELL.AS",
    "INGA.AS",
    "PHIA.AS",  # Philips
    "AD.AS",  # Ahold Delhaize
    # France
    "MC.PA",  # LVMH
    "OR.PA",  # L'Oréal
    "TTE.PA",  # TotalEnergies
    "SAN.PA",  # Sanofi
    "AIR.PA",  # Airbus
    "BNP.PA",  # BNP Paribas
    "CS.PA",  # AXA
    "DG.PA",  # Vinci
    "AI.PA",  # Air Liquide
    "SU.PA",  # Schneider Electric
    "CAP.PA",  # Capgemini
    "KER.PA",  # Kering
    "RI.PA",  # Pernod Ricard
    "SGO.PA",  # Saint-Gobain
    "VIE.PA",  # Veolia
    # Germany
    "SAP.DE",
    "SIE.DE",  # Siemens
    "ALV.DE",  # Allianz
    "MBG.DE",  # Mercedes-Benz
    "BMW.DE",
    "BAYN.DE",  # Bayer
    "EOAN.DE",  # E.ON
    "DTE.DE",  # Deutsche Telekom
    "MUV2.DE",  # Munich Re
    "IFX.DE",  # Infineon
    "DB1.DE",  # Deutsche Börse
    "HEI.DE",  # HeidelbergCement (now Heidelberg Materials)
    # Italy
    "ENEL.MI",
    "ENI.MI",
    "ISP.MI",  # Intesa Sanpaolo
    "UCG.MI",  # UniCredit
    "STM.MI",  # STMicroelectronics
    # Spain
    "IBE.MC",  # Iberdrola
    "ITX.MC",  # Inditex (Zara)
    "SAN.MC",  # Banco Santander
    "BBVA.MC",
    # Finland
    "NOKIA.HE",
    # Ireland
    "CRH.L",  # CRH (listed London)
    # Denmark
    "NOVO-B.CO",  # Novo Nordisk
    # Switzerland (ikke officielt EuroStoxx men store europæiske caps)
    "NESN.SW",  # Nestlé
    "ROG.SW",  # Roche
    "NOVN.SW",  # Novartis
    "ABBN.SW",  # ABB
    "ZURN.SW",  # Zurich Insurance
]
