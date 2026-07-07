# Databricks notebook source
import json
import random
import uuid
import os
import re

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = "synthetic_tax_dual_encoder_v3_10K"
RAW_DOC_DIR = f"{OUTPUT_DIR}/raw_docs"
PAIR_DIR = f"{OUTPUT_DIR}/pairs"

N_SAMPLES = 300     # medium synthetic build
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2     # remainder = test split

LANGUAGES = [
    "en","nl","fr","de","es","it","pt","pl",
    "sv","no","fi","ja","zh","ko"
]

# ============================================================================
# BASE ASSET TEXTS
# ============================================================================

ASSET_1 = """
Status  The Finance Act, 2019 and the Companies Income Tax (Signiﬁ cant Economic Presence) Order, 2020 expanded the scope of taxation of non-resident companies (NRCs) performing digital services in Nigeria.    NRCs deriving income from digital services are deemed to derive income from Nigeria to the extent that such NRCs have a signiﬁ cant economic presence (SEP) in the country.  NRCs deemed to have a SEP in Nigeria are required to register for taxes and to comply with the relevant income tax ﬁ ling and payment obligations in Nigeria.  The Finance Act 2021 provided that non-resident companies liable to tax on proﬁ ts arising from digital goods and services under the SEP rule may be assessed on fair and reasonable percentage of turnover if there is no assessable proﬁ t, the assessable proﬁ t is less than expected or the assessable proﬁ t cannot be ascertained. Scope Foreign companies undertaking the following activities are deemed to have a SEP in Nigeria:  Category 1 – A foreign company using digital platforms to derive gross income equal to or greater than N25 million (or its equivalence in other currencies) in a year of assessment, from any of the following activities (or combination thereof):   Streaming, or downloading services of digital contents to any person in Nigeria  Transmission of data collected about Nigerian users, which has been generated from such user’s activities on a digital interface, including a website or mobile application. 
 Provision of goods or services directly or indirectly to Nigerians through digital platforms.  Provision of intermediation services through digital platforms that link suppliers and customers in Nigeria.  Category 2 – A foreign company that uses a Nigerian domain name (.ng) or registers a website address in Nigeria. 
 Category 3 – A foreign company that has a purposeful and sustained interaction with persons in Nigeria by customizing its digital platform to target persons in Nigeria or 
reﬂ ecting the prices of its products, services or options of billing or payment in the local currency, Naira.  Reference : "https://www.oecd.org/content/dam/oecd/en/topics/policy-sub-issues/transfer-pricing/transfer-pricing-country-profile-angola.pdf" 
Rate Corporate income tax at 30% of taxable proﬁ ts. Thresholds N25 million (approximately US$26,000) for Category 1 transactions 
Exclusions Foreign companies covered under any multilateral/consensus agreement to address tax challenges arising from digitalization of the economy to which Nigeria is a party, to the extent that such agreement is effective.  So far, Nigeria has not signed up for BEPS 2.0. 
"""

ASSET_2 = """
A Gibraltar subsidiary taxed at 15% earns hash power trading income. Capital gains are exempt.
Even small exempt BTC-to-USD gains may push Dutch ETR below 10%, failing the subject
to tax test. Offshore regime elections at 0% create cliff effects. A crypto ETR control framework
is required to maintain ≥10% Dutch-standard taxation.
"""

ASSET_3 = """
On 21 February 2025, US President Trump signed a Presidential Memorandum directing a review and possible renewal of investigations into countries that have implemented 
DSTs. The memorandum speciﬁ cally targets seven countries: Austria, Canada, France, Italy, Spain, Türkiye and the UK. It also directs his administration to identify policies of 
other nations that may discriminate against US companies or impose burdens on US digital commerce and recommend actions to counteract such policies. Further on 20 
February 2025, the US Trade Representative opened a public comment period for feedback under the America First Trade Policy Presidential Memorandum and the Reciprocal 
Trade and Tariffs Presidential Memorandum.
"""

ASSET_4 = """
We have been requested to analyze the applicability of the Dutch participation exemption on CryptoCom 
NV’s (‘CryptoCom’) interest in MiningCom Ltd (‘MiningCom’). In this memorandum we will outline the 
framework of the Dutch participation exemption rules and make an initial assessment of the applicability 
of the exemption.1 We note, however, that additional information is still needed and validation is required 
to come to more tangible conclusions.Please note that we have based our analysis on information and documents provided by you. We have 
stated our understandings and assumptions below. Should any of the facts, circumstances or assumptions 
be different than stated, please let us know as this could alter the conclusion of our analysis. Also, our 
memorandum is based on our interpretation of the tax legislation and case law as per the date of this 
memorandum and is not a formal tax opinion, nor should any advice contained herein be construed as a 
formal tax opinion. 
"""

ASSET_5 = """The transfer pricing operating model allocates a cost-plus remuneration to the mining branches, 
whereas MiningCom is entitled to the profits pertaining to its hashing power trading activities; 
The Gibraltar corporate tax rate is a flat rate of 15%. Furthermore, capital gains are not taxed in 
Gibraltar. Capital losses are therefore also not deductible; Gibraltar also has an ‘off-shore tax regime’, based on which foreign income may be taxed against a corporate tax rate of 0%. MiningCom is considering whether they can also engage in activities that 
qualify under the ‘off-shore regime’ (i.e. procuring the ‘hash power’ from its group companies); 
Currently, MiningCom only has ‘on-shore activities’ which will be taxed against 15% as of financial 
year 21/22; The main assets on the balance of MiningCom consist of active trading income (i.e. USD cash), 
intellectual property (‘IP’) related to knowledge/know-how, and short-termtrading / cashpool 
intercompany receivables (with an interest charged of 4%). These assets should be allocated as 
part of the on-shore tax regime and taxed against 15%; 
MiningCom aims to keep its balance as light as possible. The bitcoins generated from the supply of 
hash power are converted to USD as soon as they are received and ideally on the same day; 
Therefore, there are generally no bitcoins on the balance sheet, but rather USD cash; 
MiningCom uses USD its functional tax currency in Gibraltar and (almost) all cost are in USD and 
most of its transactions are settled in USD; 
The excess cash of MiningCom is reserved for dividends and cash pooling. The cash needed for 
operations or the purchase of miners remains on the balance sheet (‘acquisition funds’); 
MiningCom does not derive any capital gains on the bitcoins (or only neglible gains), nor FX 
results;  The company does not invest with or in bitcoins; """

ASSET_6 = """The IP could qualify as a free portfolio investment if it is not required in the course of the business of the 
entity owning the assets. We understand the IP relates to knowledge/know-how on mining activities. 
Hence, it could be argued that the IP is not a free portfolio investment for MiningCom. We note that if the IP would be licensed within the CryptoCom group, it could qualify as a deemed free portfolio investment for the purpose of the asset test. We understand – however – that no intercompany licensing activities take place. Furthermore, we note that – even if the IP would be a (deemed) free portfolio investment – it would be taxed against the effective tax rate of 15% in accordance with Dutch standards. If the IP would be a free 
portfolio investment, it might therefore not be considered as ‘low-taxed’ for the purpose of the asset 
test. This is under the assumption that the IP would qualify under the on-shore regime.  
Lastly, we note that even if the IP would qualify as a low-taxed free portfolio investment, MiningCom 
could still meet the asset test if the IP would make up <50% of its balance sheet. This should be further 
analyzed – if required – once the balance sheet of MiningCom becomes available. 
We note that however – due to the very light balance sheet of MiningCom after the (mandatory) 
elimination of the intercompany receivables – having to rely on the asset test for the application 
exemption does not seem preferred. If there is any ‘excess’ USD cash on the balance sheet of MiningCom 
(not directly required for operations/acquisitions) this may negatively impact the asset test. Hence, this"""

ASSET_7 = """

The UK DST applies from 1 April 2020 and is payable annually, nine months after relevant accounting period. The legislation is included in Finance Act 2020, which received 
Royal Assent on 22 July 2020.  The UK tax authority published its DST manual on 19 March 2020, which explains the structure and details of the UK DST.  This manual includes 
what is meant by digital services activity and revenue, deﬁ nitions of a user and identifying revenue of UK users, detail on the role and responsibilities of the responsible member, 
as well as further details of the administration and compliance framework that applies for DST.  There have been updates to the manual since its publication, including the 
deﬁ nition of online services, the compliance framework and the list of countries that have taxes that are considered to be similar to the UK DST for the purposes of cross-border 
relief. In March 2021, HMRC made further changes to the DST manual, introducing a section on the compliance framework, updating the guidance on submitting returns for 
groups with non-GBP consolidated accounts and adding Spain to its list of countries with similar DST (for which cross-border relief would be allowed).   
Further, from 14 June 2022, the Malaysian Service Tax on Digital Services by Foreign Service Providers is no longer considered by HMRC to be similar to the UK DST for the 
purposes of cross-border relief.  Any claims for cross-border tax relief made before 14 June 2022 will be honored, but no new claims for relief relating to this tax will be accepted. 
In August 2024, HMRC updated the DST manual to conﬁ rm that for the purposes of DST cross-border relief, HMRC considers the Canadian DST to be similar to the UK DST. 
Rate 2% of gross revenues from UK in-scope activities above threshold; however, taxpayers may apply an alternative calculation method calculated based on operating margin in respect of 
in-scope activities where they are loss- making or have a very low proﬁ t margin Thresholds £500m revenues from in-scope activities provided globally and £25m of revenue from in-scope activities provided to UK users per 12-month accounting period. The ﬁ rst £25m of revenues is not subject to the tax.  £500m and £25m thresholds are applied to total revenues arising to a group from in-scope activities, rather than on an activity-by-activity basis. The group upon which the thresholds are tested is determined by reference to accounting consolidation principles. Exclusions Provision of an online marketplace by a ﬁ nancial services provider where upwards of 50% of revenues relate to the creation/trading of ﬁ nancial assets 
Effective date 1 April 2020 Reference Links as below EY Global Tax Alerts 
US initiates review of other countries' imposition of DSTs on US companies and opens comment period on nonreciprocal trade arrangements (25 February 2025) 
Six country Joint Statement on transitional approach to existing unilateral measures during period before Pillar One is in effect (25 October 2021) 
USTR releases ﬁ ndings of Section 301 investigation on DST regimes of Austria, Spain and the UK, and 301 ﬁ ndings on Vietnam’s currency valuation practices | EY – Global 
(21 January 2021)  USTR proposes 25% punitive tariff on Austrian, Indian, Italian, Spanish, Turkish and UK origin goods in response to each country’s DST; Terminates investigations for Brazil, Czech Republic, EU and Indonesia | EY – Global (29 March 2021)  USTR initiates investigations into DSTs either adopted, or under consideration, by certain jurisdictions (4 June 2020) UK releases draft clauses and guidance on Digital Services Tax (12 July 2019) UK proposes Digital Services Tax: unilateral measure announced in Budget 2018 (5 November 2018) 
USTR announces 25% punitive tariffs on six speciﬁ c countries in response to their DSTs; Suspends tariffs for 180 days (4 June 2021) """

ASSET_8="""Status  The Finance Act, 2019 and the Companies Income Tax (Signiﬁ cant Economic Presence) Order, 2020 expanded the scope of taxation of non-resident companies (NRCs) 
performing digital services in Nigeria.    NRCs deriving income from digital services are deemed to derive income from Nigeria to the extent that such NRCs have a signiﬁ cant economic presence (SEP) in the country.  NRCs deemed to have a SEP in Nigeria are required to register for taxes and to comply with the relevant income tax ﬁ ling and payment obligations in Nigeria.  The Finance Act 2021 provided that non-resident companies liable to tax on proﬁ ts arising from digital goods and services under the SEP rule may be assessed on fair and 
reasonable percentage of turnover if there is no assessable proﬁ t, the assessable proﬁ t is less than expected or the assessable proﬁ t cannot be ascertained. 
Scope Foreign companies undertaking the following activities are deemed to have a SEP in Nigeria: 
 Category 1 – A foreign company using digital platforms to derive gross income equal to or greater than N25 million (or its equivalence in other currencies) in a year of 
assessment, from any of the following activities (or combination thereof):  
 Streaming, or downloading services of digital contents to any person in Nigeria  Transmission of data collected about Nigerian users, which has been generated from such user’s activities on a digital interface, including a website or mobile application.  Provision of goods or services directly or indirectly to Nigerians through digital platforms. 
 Provision of intermediation services through digital platforms that link suppliers and customers in Nigeria. 
 Category 2 – A foreign company that uses a Nigerian domain name (.ng) or registers a website address in Nigeria. 
 Category 3 – A foreign company that has a purposeful and sustained interaction with persons in Nigeria by customizing its digital platform to target persons in Nigeria or 
reﬂ ecting the prices of its products, services or options of billing or payment in the local currency, Naira.  Rate Corporate income tax at 30% of taxable proﬁ ts. 
Thresholds N25 million (approximately US$26,000) for Category 1 transactions Exclusions Foreign companies covered under any multilateral/consensus agreement to address tax challenges arising from digitalization of the economy to which Nigeria is a party, to the extent that such agreement is effective.  So far, Nigeria has not signed up for BEPS 2.0. """

ASSET_9="""Status  Currently Mexico does not impose a DST (December 2021). 
 Effective as of 2022, Mexico City has a contribution on deliveries (e.g., food, parcels) through digital platforms. The new tax is equal to 2% of the total charge before taxes for 
each delivery made through ﬁ xed or mobile devices that allow users to contract for the delivery of parcels, food, provisions, or any type of merchandise delivered in Mexico City’s 
territory. This tax is to be paid by the platform and cannot be transferred to the clients or persons making the delivery. 
 The tax authority (Servicio de Administración Tributaria) published the Miscellaneous Fiscal Resolution which includes rules and guidance on the remission of withholding tax by 
foreign digital service providers.  In 2018, a DST Bill was submitted to the Mexican Congress to apply a 3% tax on the revenue of digital providers that are residents in Mexico or that have a permanent establishment in the country. The Bill was not approved by the Congress.  As of March 2021, a 16% VAT is applicable on digital services provided by foreign residents with no permanent establishment in Mexico when the recipient of the service is 
located in Mexico. This tax applies to certain digital services such as providing access to content for users, gaming and learning; the law also applies to platforms providing 
intermediation services. The foreign digital supplier is obligated to meet several compliance and disclosure obligations before the Mexican tax authorities. These obligations 
include, but are not limited to, registering in Mexico, reporting and emitting tax on a monthly basis and providing certain disclosures as to services provided in Mexico 
 In January 2024, the tax authorities published a list of almost 201 foreign digital service providers registered before the Mexican tax authorities. 
Scope  Mexico City has a contribution on deliveries (e.g., food, parcels) through digital platforms for the delivery of parcels, food, provisions, or any type of merchandise delivered in the Mexico City territory. This tax is paid by the platform and cannot be transferred to the clients or persons making the delivery. 
 VAT is applicable on digital services provided by foreign residents with no permanent establishment in Mexico when the recipient of the service is located in Mexico. This tax 
applies to certain digital services such as providing access to content for users, gaming and learning; the law also applies to platforms providing intermediation services 
Rate  Mexico City contribution on deliveries 2%  VAT 16% Thresholds N/A Exclusions N/A """

ASSET_10="""Status DST 
 Inland Revenue Department has issued “Procedures relating to digital service tax, 2079 (2022)” in exercise of powers conferred by Sub-section (5) of section 20 of Finance Act, 
2079 pursuant to which a non-resident person providing taxable digital services to consumers in Nepal is required to pay Digital Services Tax (DST) subject to a speciﬁ ed this is long version of the document detailing the digital service tax threshold.  VAT 
 Inland Revenue Department has issued “Procedure relating to value added tax on digital service provided by non-resident person
threshold. VAT The Inland Revenue Department has issued “Procedure relating to value added tax on digital service provided by non-resident person, 2079 (2022)”, in exercise of powers 
conferred by Section 10b1, Sub-section (1b) of section 18 and Sub-section (7b) of Section 19 of Value Added Tax Act, 2052 (1995), pursuant to which a non-resident person 
providing taxable digital services to consumers in Nepal are liable to collect and pay VAT subject to a speciﬁ ed threshold.  
After the Nepal Budget announced in May 2024, there has been a change to the DST and VAT thresholds for digital services provided by non-residents to consumers in Nepal. 
Scope DST The DST covers digital services whose delivery requires the use of information technology and are provided automatically through the internet with minimal human intervention. 
These services include (i) advertisements; (ii) cloud services; (iii) data storage services; (iv) e-books, e-libraries, e-newspapers; (v) education, consultancy, skill development and 
training services; (vi) downloads of data, images and similar services; (vii) gaming services; (viii) movies, television, music, over-the-top and other similar subscription based 
services; (ix) online marketplace services and goods and services to be provided through it; (x) sales of data collected from Nepalese residents; (xi) services related to mobile 
applications; (xii) supply and updates of software; and (xiii) other similar services.  
 A non-resident person subject to DST is required to register and obtain a taxpayer identiﬁ cation number and must ﬁ le a tax return and pay tax online within three months of the 
completion of an income year, otherwise penalties apply. VAT  A non-resident person subject to VAT is required to register under the VAT laws, issue an electronic invoice for each sale and ﬁ le a tax return and pay tax online otherwise penalties may apply.  Every taxable person must ﬁ le a VAT return for each tax period, by the 25th day of the following month. Where annual taxable turnover is greater than NPR10 million in a ﬁ nancial year (as per the Nepali calendar1), the ﬁ ling obligation is monthly. Where annual taxable turnover is equal to or less than NPR10 million, the ﬁ ling obligation is four-monthly (i.e., the returns are ﬁ led every four months). The tax period in Nepal is the calendar month. 
 Digital services provided to persons other than taxable persons are subject to VAT under the normal reverse charge mechanism as the same would qualify as an import of service. Disclaimer : OECD Guidelines for Multinational Enterprises (OECD Guidelines) are not applicable to the above"""

ASSET_11="""Software deployment with DE44DEUTDEUT1234567890 and operations: The release pipeline triggers only after unit and integration tests pass in continuous integration. A blue-green deployment strategy reduces downtime during application cutover. Health probes from the load balancer return 200 OK across all replicas before traffic shifts. We rolled back the canary when the error budget exceeded the defined threshold. Observability relies on distributed tracing, structured logs, and metrics correlated to SLOs. Feature flags gate risky changes to minimize blast radius and enable fast toggles. The runbook documents rollback steps and database migration sequencing. Incident response includes paging on-call, updating the status page, and triage. Capacity planning forecasts CPU, memory, and IOPS for expected peak periods. Postmortems capture root causes, contributing factors, and preventative actions to avoid recurrence."""

ASSET_12="""Healthcare clinic operations: The appointment system schedules new patient intakes and routine follow-ups each week. Nurses with ID ID. DL1234567 triage cases based on urgency, symptoms, and vital sign deviations. Inventory checks maintain accurate counts of PPE and medication stock levels. Electronic records such as passport numbers starting with K8787937 capture vitals, allergies, and current prescription details. The lab transmits results securely with a target turnaround time under twenty-four hours. Infection control protocols require sanitization of rooms after each visit. Telehealth sessions follow privacy guidelines and informed consent procedures. Shift handoffs use standardized notes to reduce communication gaps during transitions. The pharmacy coordinates refills, validates dosages, and checks contraindications. Community outreach promotes preventative screenings and general wellness programs."""

ASSET_13="""Supply chain and logistics: The warehouse management system assigns bin locations for incoming goods efficiently. Barcode scans update inventory counts in real time and reduce manual entry errors. The routing engine optimizes delivery paths using live traffic and carrier constraints. Cold chain monitoring records temperature data across transport milestones and alerts deviations. Clearance paperwork is prepared thoroughly to avoid border delays and rechecks. Vendors share forecasts to align production runs with shipment cycles and demand. Dock scheduling staggers arrivals to prevent yard congestion and idle time. Reverse logistics processes returns, refurbishments, and restocking with minimal waste. Demand planning models include seasonality and lead time variability considerations. Performance indicators track on-time delivery rates, pick accuracy, and dock-to-stock times."""

ASSET_14="""Cybersecurity practices: Multi-factor authentication is enforced for privileged and administrative accounts consistently. Least privilege access policies restrict lateral movement and reduce attack surface. Endpoint detection and response tools flag suspicious process spawn chains promptly. Patch cycles remediate high-severity vulnerabilities according to prioritization policies. Secrets management rotates keys, audits vault access events, and eliminates hardcoding. Network segmentation isolates critical systems from general traffic and external exposure. Phishing simulations train staff to recognize deceptive content and report incidents. Backup encryption protects restore points and ensures integrity during recovery. Threat modeling reviews attack surfaces before major releases and architectural changes. Incident playbooks define containment, eradication, and recovery sequences with clear roles. K1234567. 123-45-6789. 4111 1111 1111 1111. 01/23/1990. TIN. DL1234567"""


ASSET_15="""Museum exhibition management: Curators develop themes and select works to form coherent narratives for visitors with IBAN . Conservation assesses material stability and environmental requirements for display. Loan agreements detail conditions, insurance coverage, and transport documentation precisely. Exhibit lighting is calibrated to preserve pigments, reduce glare, and enhance visibility. Visitor flow design prevents bottlenecks and guides exploration through the gallery. Interactive kiosks provide context using audio explanations and visual references. Docent training covers artist backgrounds, historical context, and interpretive frameworks. Marketing produces posters, social media content, and newsletters to promote attendance. Ticketing systems track daily counts, bookings, and peak visiting hours accurately. Post-exhibit reviews evaluate engagement levels and educational outcomes for future planning. K1234567. 123-45-6789. 4111 1111 1111 1111. 01/23/1990. ID. DL1234567."""

ASSET_POOL = [ASSET_1,ASSET_4, ASSET_5, ASSET_6, ASSET_7, ASSET_8, ASSET_9, ASSET_10, ASSET_11, ASSET_12, ASSET_13, ASSET_14, ASSET_15]

# ============================================================================
# LABEL SPACE
# ============================================================================

ALL_LABELS = [
    "tax_problem","tax_solution","tax_type","tax_topic","year","client_addressed",
    "internal_email","final_document","draft_document","long_document","short_email",
    "has_disclaimer","has_advisory_structure","has_sow_reference","has_citations",
    "has_appendices","contains_numbers","pii_present","contains_board_resolution",
    "contains_financials"
]

# ============================================================================
# REAL-WORLD PII REGEX PATTERNS
# ============================================================================

PII_REGEX = {
    "passport": r"\b[A-PR-WYa-pr-wy][0-9]{7}\b|\bpassport\b",
    "iban": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "dob": r"\b(?:\d{2}[/-]\d{2}[/-]\d{4})\b",
    "tin": r"\b(?:tax identification number|TIN)\b",
    "driver_license": r"\b(driver'?s license|DL[0-9]{6,})\b"
}

SEVERITY_WEIGHT = {
    "passport": 0.6,
    "iban": 0.5,
    "ssn": 0.8,
    "credit_card": 1.0,
    "dob": 0.4,
    "tin": 0.5,
    "driver_license": 0.4
}

def detect_real_pii(text):
    """
    Returns:
        pii_flag  (0/1)
        pii_types (list of detected categories)
        pii_score (0–1 severity-weighted, capped)
    """
    found = []

    for pii_type, pattern in PII_REGEX.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(pii_type)

    if not found:
        return 0, [], 0.0

    # severity score
    raw_score = sum(SEVERITY_WEIGHT.get(t, 0.3) for t in found)
    pii_score = min(raw_score, 1.0)

    return 1, found, pii_score

# ============================================================================
# TEXT AUGMENTATION
# ============================================================================

def generate_label_subset():
    k = random.randint(1, 4)
    return random.sample(ALL_LABELS, k)

def maybe_add_numbers(text):
    if random.random() < 0.25:
        return text + f" The transaction amount was {random.randint(5000, 5000000)} USD."
    return text

def add_language_noise(text, lang):
    prefix = {
        "nl": "Volgens Nederlandse fiscale regels: ",
        "fr": "Selon la réglementation fiscale: ",
        "de": "Nach deutschem Steuerrecht: ",
        "es": "Según la normativa fiscal: ",
        "it": "Secondo la normativa fiscale: ",
        "pt": "De acordo com as regras fiscais: ",
        "pl": "Zgodnie z przepisami podatkowymi: ",
    }
    return prefix.get(lang, "") + text

def maybe_add_structural_elements(text, labels):
    if "has_disclaimer" in labels:
        text += " Our advice is based on current OECD tax legislation and subject to change."
    if "has_citations" in labels:
        text += " See Art. 13 CITA and relevant Kluwer commentary."
    if "has_appendices" in labels:
        text += " Appendix A includes supporting calculations."
    if "contains_board_resolution" in labels:
        text += " Board resolution dated 2022-03-11 is attached."
    if "contains_financials" in labels:
        text += " Financial statements indicate OPEX of USD 4.3m."
    return text

def random_year_tag():
    return random.choice(["2021", "2022", "FY21/22", "2020/21", "AY2022"])

# ============================================================================
# DOCUMENT GENERATION
# ============================================================================

def build_document():
    lang = random.choice(LANGUAGES)
    base = random.choice(ASSET_POOL)
    labels = generate_label_subset()

    text = base.strip()
    text = add_language_noise(text, lang)
    text = maybe_add_numbers(text)
    text = maybe_add_structural_elements(text, labels)

    # ---- Real-world PII injection (optional) ----
    if random.random() < 0.20:
        text += f" Passport number: {random.randint(10000000, 99999999)}."

    if "year" in labels:
        text += f" This analysis applies to {random_year_tag()}."

    # Detect PII using regex
    pii_flag, pii_types, pii_score = detect_real_pii(text)

    # Maintain consistency with pii_present label
    if "pii_present" in labels and pii_flag == 0:
        labels.remove("pii_present")
    elif "pii_present" not in labels and pii_flag == 1:
        labels.append("pii_present")

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "language": lang,
        "label": labels,
        "pii_flag": pii_flag,
        "pii_types": pii_types,
        "pii_score": pii_score
    }

# ============================================================================
# PAIRWISE LABEL DEFINITIONS
# ============================================================================

LABEL_DESCRIPTIONS = {
   "tax_problem": (
        "A document that discusses a tax issue, error, dispute, challenge, "
        "or risk requiring attention. Often includes facts, circumstances, "
        "concerns, or problems that need resolution."
    ),
     "tax_solution": (
        "A document that proposes a tax recommendation, remediation, strategy, "
        "or solution to a tax-related problem. Typically contains actionable "
        "guidance, analysis, or next steps."
    ),
    "tax_type": (
        "A document that specifies or focuses on a category of tax, such as income tax, "
        "corporate tax, VAT, withholding tax, transfer pricing, excise, or any other tax classification."
    ),
    "tax_topic": (
        "A document that centers on a particular tax topic or theme. The text discusses "
        "a specific subject area within taxation, such as regulations, rulings, compliance, or policy topics."
    ),
    "year": (
        "The document explicitly references a tax year or financial year (e.g., 2020, FY22, assessment year). "
        "May reference filing periods, statutory years, or year-specific positions."
    ),
    "client_addressed": (
        "The document is addressed to a client, external party, partner, or someone outside the internal team. "
        "May contain greetings, sign-offs, or language appropriate for external communication."
    ),
    "internal_email": (
        "The document is an internal communication. It is written for colleagues and not intended for external clients. "
        "Usually informal and operational in tone."
    ),
    "final_document": (
        "The document is a final version, polished draft, or signature-ready deliverable. "
        "May include a title page, signature block, or formal structure."
    ),
    "draft_document": (
        "The document is a draft or work-in-progress version. It may be incomplete, unpolished, or explicitly labeled as a draft."
    ),
    "long_document": (
        "The document is lengthy or contains substantial written content. Typically more detailed, possibly containing multiple sections, pages, or long explanations."
    ),
    "short_email": (
        "The document is a very short email, generally one sentence or extremely brief. Often lacks analysis, structure, or substantial content."
    ),
    "has_disclaimer": (
        "The document contains a disclaimer such as ‘our advice is based on …’, limitations of liability, scope restrictions, conditional language, or professional disclaimers."
    ),
    "has_advisory_structure": (
        "The document follows a structured advisory format: introduction, executive summary, facts or circumstances, analysis, and conclusion. It resembles a formal deliverable."
    ),
    "has_sow_reference": (
        "The document references an engagement letter, agreement, SOW (statement of work), or similar contract. Often contains wording like ‘based on our agreement dated…’."
    ),
    "has_citations": (
        "The document contains citations, references, footnotes, case law, regulatory sources, or supporting documentation."
    ),
    "has_appendices": (
        "The document includes appendices or additional supporting materials at the end."
    ),
    "contains_numbers": "Contains amounts, figures or numeric computations.",
    "pii_present": "Contains non-trivial PII such as passport, IBAN, SSN.",
    "contains_board_resolution": "Mentions board resolutions.",
    "contains_financials": "Mentions financial statements or financial metrics."
}

# ============================================================================
# PAIR BUILDER
# ============================================================================

def make_pairs(doc):
    pairs = []
    true_labels = set(doc["label"])
    doc_text = doc["text"]

    for label in ALL_LABELS:
        label_text = LABEL_DESCRIPTIONS[label]

        # ---- semantic target ----
        target = 1 if label in true_labels else 0

        # ---- soft-score influenced by #PII items (C) ----
        if target == 1:
            # semantic + PII-weighted influence
            soft = 0.75 + (doc["pii_score"] * 0.25)
        else:
            soft = random.uniform(0.0, 0.25)

        pairs.append({
            "doc_id": doc["id"],
            "doc_text": doc_text,
            "label_name": label,
            "label_text": label_text,
            "target": target,
            "soft_score": float(round(soft, 4)),
            "pii_flag": doc["pii_flag"],
            "pii_score": float(doc["pii_score"]),
            "pii_types": doc["pii_types"]
        })

    return pairs

# ============================================================================
# BUILD & SAVE DATASET
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DOC_DIR, exist_ok=True)
os.makedirs(PAIR_DIR, exist_ok=True)

all_docs = [build_document() for _ in range(N_SAMPLES)]
random.shuffle(all_docs)

n_train = int(TRAIN_RATIO * N_SAMPLES)
n_val = int(VAL_RATIO * N_SAMPLES)

splits = {
    "train": all_docs[:n_train],
    "val": all_docs[n_train:n_train+n_val],
    "test": all_docs[n_train+n_val:]
}

# SAVE RAW DOCS
for split, docs in splits.items():
    with open(f"{RAW_DOC_DIR}/{split}.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

# SAVE PAIRWISE NLI DATA
for split, docs in splits.items():
    with open(f"{PAIR_DIR}/{split}.jsonl", "w") as f:
        for d in docs:
            for row in make_pairs(d):
                f.write(json.dumps(row) + "\n")

print("[INFO] Dataset created successfully!")
print(f"[INFO] RAW DOCS:   {RAW_DOC_DIR}")
print(f"[INFO] PAIRWISE:   {PAIR_DIR}")
