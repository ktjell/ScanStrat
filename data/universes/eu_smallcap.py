"""data/universes/eu_smallcap.py

Ca. 200 likvide europaeiske small/mid-cap tickers (yfinance-format).

Daekker DAX, MDAX, TecDAX, CAC Mid/Small, AEX small, OMX Nordic Mid,
FTSE 250 (top likvide) og SMI Expanded.

Suffixes:
    .DE = XETRA (Frankfurt)
    .PA = Euronext Paris
    .AS = Euronext Amsterdam
    .MI = Borsa Italiana
    .MC = Bolsa Madrid
    .ST = Stockholm
    .CO = Koebenhavn
    .OL = Oslo
    .HE = Helsinki
    .L  = London (GBp)
    .SW = Zurich
"""

from __future__ import annotations


def get_eu_smallcap_tickers() -> list[str]:
    """Returner ca. 200 likvide europaeiske small/mid-cap tickers."""
    return list(_EU_SMALLCAP)


_EU_SMALLCAP: list[str] = [
    # -----------------------------------------------------------------------
    # Tyskland — MDAX / TecDAX / SDAX
    # -----------------------------------------------------------------------
    "AIXA.DE",  # AIXTRON — halvledere
    "EVT.DE",  # Evotec — biotech
    "NEM.DE",  # Nemetschek — software (AEC)
    "CGMK.DE",  # CompuGroup Medical (COP.DE var forkert)
    "GXI.DE",  # Gerresheimer — farmaceutisk emballage
    "JEN.DE",  # Jenoptik — fotonik/forsvar
    "KGX.DE",  # KION Group — gaffeltrucks
    "NDX1.DE",  # Nordex — vindenergi (NDXG.DE var forkert)
    "PUM.DE",  # PUMA — sportstoj
    "RAA.DE",  # RATIONAL AG — koekkenteknik
    "S92.DE",  # SMA Solar Technology
    "SDF.DE",  # K+S — kemikalier/mineraler
    "TKA.DE",  # thyssenkrupp
    "VBK.DE",  # Verbio — biobraendstof
    "WAF.DE",  # Siltronic — halvledermaterialer
    "WCH.DE",  # Wacker Chemie
    # ZO1.DE (Zooplus) — aflistet efter opkoeb 2022
    "FNTN.DE",  # freenet AG — telekommunikation
    "DWS.DE",  # DWS Group — kapitalforvaltning
    "LEG.DE",  # LEG Immobilien — fast ejendom
    # TAG.DE — ingen data (aflistet/fusioneret)
    "AT1.DE",  # Aroundtown — fast ejendom (ARR.DE var forkert)
    # GKPRF.DE (GK Software) — aflistet efter opkoeb
    "BDT.DE",  # Bertrandt — ingeniortjenester
    "ECK.DE",  # Eckert & Ziegler — medicinsk isotoper
    "GIL.DE",  # Gildemeister (DMG MORI)
    "TLX.DE",  # Talanx — forsikring
    "HNR1.DE",  # Hannover Ruck — genforsikring (mid cap)
    "MBB.DE",  # MBB Industries
    "BOSS.DE",  # HUGO BOSS
    "HLE.DE",  # Hella — bildele
    "KWS.DE",  # KWS SAAT — froavl
    # 1COV.DE (Covestro) — aflistet efter opkoeb af ADNOC 2024
    "SIX2.DE",  # Sixt — biludlejning
    "SY1.DE",  # Symrise — aromastoffer (mid)
    "DHL.DE",  # DHL Group (DPW.DE var gammelt symbol)
    # -----------------------------------------------------------------------
    # Frankrig — CAC Mid 60 / SBF 120
    # -----------------------------------------------------------------------
    "SOI.PA",  # Soitec — SOI-wafers
    "WAVE.PA",  # Wavestone — IT-raadgivning
    "TRI.PA",  # Trigano — campingvogne
    "SWP.PA",  # Sword Group — IT
    "ALO.PA",  # Alstom — tog/transport (ALSTOM.PA var forkert)
    "GTT.PA",  # GTT — LNG containere
    "LR.PA",  # Legrand — elektrisk infrastruktur (mid)
    # NEOEN.PA — aflistet efter opkoeb af AGL Energy 2024
    "APAM.PA",  # Aperam — specialstaal
    "SU.PA",  # Schneider Electric (mid/large)
    "RCO.PA",  # Remy Cointreau
    "MRX.PA",  # Mersen — grafit/termisk (MERY.PA var forkert)
    "DIM.PA",  # Sartorius Stedim Biotech (SK.PA var forkert)
    "VLA.PA",  # Valneva — vacciner
    "FRVIA.PA",  # FORVIA (ex-Faurecia)
    "GFC.PA",  # Gecina — fast ejendom
    "KER.PA",  # Kering (luxury, mid)
    "VIE.PA",  # Veolia Environnement
    "GNFT.PA",  # GENFIT — biotech (GENFIT.PA var forkert)
    "INF.PA",  # Infotel (INFE.PA var forkert)
    "HO.PA",  # Thales (forsvar, mid)
    # -----------------------------------------------------------------------
    # Holland — AEX small / AMX
    # -----------------------------------------------------------------------
    "BESI.AS",  # BE Semiconductor — packaging
    "ASM.AS",  # ASM International — wafer processing
    "LIGHT.AS",  # Signify — belysning
    "OCI.AS",  # OCI — goedning/ammoniak
    "AALB.AS",  # Aalberts Industries — teknik
    "BRNL.AS",  # Brunel International — rekruttering
    "CTPNV.AS",  # CTP — fast ejendom
    "FAGR.AS",  # Fagron — farmaceutisk
    "FLOW.AS",  # Intertrust (nu CSC)
    "HYDRA.AS",  # Hydratec Industries
    "IMCD.AS",  # IMCD Group — kemikalier distribution
    "INPST.AS",  # InPost — pakkebokse
    "JDEP.AS",  # JDE Peet's — kaffe
    "KPN.AS",  # KPN Telecom (mid)
    "NSI.AS",  # NSI — kontorejendomme
    "NN.AS",  # NN Group — forsikring
    "PHIA.AS",  # Philips (mid)
    "RAND.AS",  # Randstad — rekruttering
    "TOM2.AS",  # TomTom
    "UMG.AS",  # Universal Music Group
    "WKL.AS",  # Wolters Kluwer (mid)
    # -----------------------------------------------------------------------
    # Sverige — OMX Mid Cap
    # -----------------------------------------------------------------------
    "LIAB.ST",  # Lindab — ventilation
    "NIBE-B.ST",  # NIBE Industrier — varmepumper
    "HUSQ-B.ST",  # Husqvarna — haveudstyr
    # SWMA.ST (Swedish Match) — aflistet efter opkoeb af PMI 2022
    "SECU-B.ST",  # Securitas — sikkerhed
    "SSAB-A.ST",  # SSAB — specialstaal
    "TREL-B.ST",  # Trelleborg — polymerteknologi
    "SWEC-B.ST",  # Sweco — ingeniortjenester
    "HUFV-B.ST",  # Hufvudstaden — fast ejendom (HUFVB.ST var forkert)
    "BALD-B.ST",  # Balder — fast ejendom
    "CAST.ST",  # Castellum — fast ejendom
    "COOR.ST",  # Coor Service Management
    "DOM.ST",  # Dometic — RV-udstyr
    "EPI-B.ST",  # Epiroc — minedrift udstyr
    "FABG.ST",  # Fabege — fast ejendom
    "GETI-B.ST",  # Getinge — medicinsk udstyr
    # HIQ.ST — aflistet efter opkoeb 2022
    "INVE-B.ST",  # Investor AB (holding)
    "KABE-B.ST",  # KABE Group — campingvogne
    "LATO-B.ST",  # Latour — industri holding
    "LUND-B.ST",  # Lundbergforetagen
    "NDA-SE.ST",  # Nordea Bank (mid)
    "NOLA-B.ST",  # Nolato — polymerteknologi (NMAN.ST var forkert)
    "SAAB-B.ST",  # Saab — forsvar
    "SKF-B.ST",  # SKF — lejer
    "SWED-A.ST",  # Swedbank
    "VOLV-B.ST",  # Volvo (trucks) (VOLVB.ST var forkert)
    # -----------------------------------------------------------------------
    # Danmark — OMX Mid / Small
    # -----------------------------------------------------------------------
    "AMBU-B.CO",  # Ambu — medicinsk udstyr
    "RBREW.CO",  # Royal Unibrew — drikkevarer
    "GN.CO",  # GN Audio/Store Nord — headsets
    "NETC.CO",  # Netcompany — IT
    "TRYG.CO",  # Tryg Forsikring
    "BAVA.CO",  # Bavarian Nordic — vacciner
    "COLO-B.CO",  # Coloplast B (mid)
    "DEMANT.CO",  # Demant — horeapparater
    "FLS.CO",  # FLSmidth — minedrift udstyr
    "HLUN-B.CO",  # H. Lundbeck — pharma
    "NNIT.CO",  # NNIT — IT
    "NTG.CO",  # NTG Nordic Transport
    "ROCK-B.CO",  # Rockwool — isolering
    "SCHB.CO",  # Schouw & Co — holding
    "SPNO.CO",  # Spar Nord Bank
    "SYDB.CO",  # Sydbank
    "VWS.CO",  # Vestas Wind Systems (mid)
    "NZYM-B.CO",  # Novonesis (ex Chr. Hansen + Novozymes)
    "ZEAL.CO",  # Zealand Pharma
    # -----------------------------------------------------------------------
    # Norge — Oslo Bors Mid
    # -----------------------------------------------------------------------
    "AFG.OL",  # AF Gruppen — byggeri
    "AKRBP.OL",  # Aker BP — olie
    "ATEA.OL",  # Atea — IT-infrastruktur
    "BEWI.OL",  # BEWi — emballage
    "BOUVET.OL",  # Bouvet — IT-raadgivning
    "ENTRA.OL",  # Entra — fast ejendom
    "FLNG.OL",  # Flex LNG
    "GOGL.OL",  # Golden Ocean — shipping
    "GSF.OL",  # Grieg Seafood (GRIEG.OL var forkert)
    "KAHOOT.OL",  # Kahoot — edtech
    "KIT.OL",  # Kitron — elektronik
    "MULCON.OL",  # Multiconsult (MHWH.OL var forkert)
    "MPCC.OL",  # MPC Containerships
    "NRC.OL",  # NRC Group — infrastruktur
    "PGS.OL",  # PGS — seismik
    "SCATC.OL",  # Scatec — sol-energi
    "SUBC.OL",  # Subsea 7
    "TGS.OL",  # TGS — seismiske data
    "VOW.OL",  # Vow — carbon capture (VWSB.OL var forkert)
    "RECSI.OL",  # REC Silicon — halvledermaterialer
    "SALM.OL",  # SalMar — laks
    "MOWI.OL",  # Mowi — laks
    "BWLPG.OL",  # BW LPG — shipping
    # -----------------------------------------------------------------------
    # Finland — OMX Helsinki Mid
    # -----------------------------------------------------------------------
    "KEMIRA.HE",  # Kemira — vandkemi
    "METSB.HE",  # Metsa Board — emballage
    "NDA-FI.HE",  # Nordea (FI-notering)
    "ORNBV.HE",  # Orion Corp — pharma
    "TIETO.HE",  # TietoEVRY — IT
    "TYRES.HE",  # Nokian Tyres
    "UPM.HE",  # UPM-Kymmene — skovbrug/kemi
    "WRT1V.HE",  # Wartsila — marine motorer
    # CGCBV.HE (Cargotec) — fusioneret; split til Cargotec + Kalmar 2023
    "KALMAR.HE",  # Kalmar — havnekraner (udskilt 2024)
    "FORTUM.HE",  # Fortum — energi
    "NESTE.HE",  # Neste — vedvarende braendstof
    "STERV.HE",  # Stora Enso — papir/emballage
    # -----------------------------------------------------------------------
    # UK — FTSE 250 (toep likvide, GBp)
    # -----------------------------------------------------------------------
    "AUTO.L",  # Auto Trader Group
    "BARC.L",  # Barclays (mid)
    "BDEV.L",  # Barratt Developments — byggeri
    "BKG.L",  # Berkeley Group — byggeri
    "BNZL.L",  # Bunzl — distribution
    "BOO.L",  # BOOHOO — fast fashion
    # BVIC.L (Britvic) — aflistet efter opkoeb af Carlsberg 2024
    "DPLM.L",  # Diploma — specialprodukter
    "EMG.L",  # Man Group — hedgefond
    "FEVR.L",  # Fevertree Drinks
    "GRG.L",  # Greggs — bageri
    "HLMA.L",  # Halma — sikkerhed/medicinsk
    "HSW.L",  # Hostelworld
    "IMI.L",  # IMI — fluidteknologi
    "INF.L",  # Informa — medier
    "JMAT.L",  # Johnson Matthey — kemi
    "MKS.L",  # Marks & Spencer
    "MRO.L",  # Melrose Industries
    "NXT.L",  # Next — mode
    "OCDO.L",  # Ocado — online dagligvarer
    "QQ.L",  # Quilter — formueforvaltning
    "RTO.L",  # Rentokil — pest control
    "SDR.L",  # Schroders — kapitalforvaltning
    # SMDS.L (DS Smith) — aflistet efter fusion med IP 2024
    "SPX.L",  # Spirax-Sarco
    "SSE.L",  # SSE — energi (mid)
    "WEIR.L",  # Weir Group — ingenior
    "WPP.L",  # WPP — reklame
    "DGE.L",  # Diageo — spiritus
    "EXPN.L",  # Experian — kreditdata
    "LGEN.L",  # Legal & General — forsikring
    "PSON.L",  # Pearson — uddannelse
    # -----------------------------------------------------------------------
    # Spanien — IBEX 35 small / BME Growth
    # -----------------------------------------------------------------------
    "ACS.MC",  # ACS — infrastruktur
    "AENA.MC",  # Aena — lufthavne
    "ALM.MC",  # Almirall — pharma
    "ENG.MC",  # Enagás — gas
    "FDR.MC",  # Fluidra — svommebassiner
    "GRF.MC",  # Grifols — blodplasma
    "IAG.MC",  # IAG (BA+Iberia) — luftfart
    "LOG.MC",  # Logista — distribution
    "MAP.MC",  # Mapfre — forsikring
    "MRL.MC",  # Merlin Properties — REIT
    "NHH.MC",  # NH Hotel Group
    "NTGY.MC",  # Naturgy Energy
    "PHM.MC",  # Pharmamar — biotech
    "REP.MC",  # Repsol — olie/gas
    "SOL.MC",  # Solaria — sol-energi
    # -----------------------------------------------------------------------
    # Italien — FTSE MIB small / STAR
    # -----------------------------------------------------------------------
    "AMP.MI",  # Amplifon — horeapparater
    "BMED.MI",  # Banca Mediolanum
    "BRE.MI",  # Brembo — bremser (BREM.MI var forkert)
    "DAN.MI",  # Danieli — staalvaerk
    "ENEL.MI",  # Enel (mid)
    "ERG.MI",  # ERG — vedvarende energi
    "IP.MI",  # Interpump Group — hydraulik
    "IG.MI",  # Italgas (IVG.MI var forkert)
    "LDO.MI",  # Leonardo — forsvar
    "MARR.MI",  # MARR — foodservice
    "MONC.MI",  # Moncler — luksusmoe
    "PIRC.MI",  # Pirelli
    "PRY.MI",  # Prysmian — kabler
    "REY.MI",  # Reply — IT (REC.MI var forkert)
    "SOL.MI",  # Sol Group — gasser
    "TEN.MI",  # Tenaris — staalror (mid)
    "TIT.MI",  # Telecom Italia
    "TOD.MI",  # Tod's — luksus
    "STLAM.MI",  # Stellantis — biler
]
