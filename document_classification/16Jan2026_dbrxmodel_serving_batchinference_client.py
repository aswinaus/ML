# Databricks notebook source
import requests
import json

token=""
url = "https://adb-156333768771.11.azuredatabricks.net/serving-endpoints/classification/invocations"
headers = {"Authorization": f"Bearer {token}"}

payload = {
    "dataframe_records": [
        {
            "doc_text": "In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide.1. Overview of the Global Health Landscape In the past decades, the world has seen repeated reminders of how vulnerable health systems can be. Outbreaks of infectious diseases, such as influenza variants, coronaviruses, viral hemorrhagic fevers, and antimicrobial-resistant pathogens, have emphasized the need for robust surveillance systems. Climate-driven events—including heat waves, floods, and vector expansion—have intensified the burden of diseases such as malaria, dengue, and cholera. Simultaneously, chronic illnesses such as diabetes, cardiovascular diseases, and mental health conditions continue to rise in prevalence, placing additional strain on already stressed health infrastructures.",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        }
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())


# COMMAND ----------

import requests
import json

# Databricks token and endpoint
token = ""  # Replace with your PAT
url = "https://adb-15633375765019771.11.azuredatabricks.net/serving-endpoints/classification/invocations"

# Headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Payload in dataframe_split format
payload = {
    "dataframe_split": {
        "columns": ["doc_text", "label_text"],
        "data": [
            [
                "In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide.1. Overview of the Global Health Landscape In the past decades, the world has seen repeated reminders of how vulnerable health systems can be. Outbreaks of infectious diseases, such as influenza variants, coronaviruses, viral hemorrhagic fevers, and antimicrobial-resistant pathogens, have emphasized the need for robust surveillance systems. Climate-driven events—including heat waves, floods, and vector expansion—have intensified the burden of diseases such as malaria, dengue, and cholera. Simultaneously, chronic illnesses such as diabetes, cardiovascular diseases, and mental health conditions continue to rise in prevalence, placing additional strain on already stressed health infrastructures.",
                "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
            ]
        ]
    }
}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Print response
print(response.status_code)
print(json.dumps(response.json(), indent=2))


# COMMAND ----------

import requests
import json

# Databricks token and endpoint
token = "" 
url = "https://adb-15667568538019771.11.azuredatabricks.net/serving-endpoints/classification/invocations"

# Headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Batch Inference Payload 
payload = {
    "dataframe_records": [
        {
            "doc_text": "In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide.1. Overview of the Global Health Landscape In the past decades, the world has seen repeated reminders of how vulnerable health systems can be. Outbreaks of infectious diseases, such as influenza variants, coronaviruses, viral hemorrhagic fevers, and antimicrobial-resistant pathogens, have emphasized the need for robust surveillance systems. Climate-driven events—including heat waves, floods, and vector expansion—have intensified the burden of diseases such as malaria, dengue, and cholera. Simultaneously, chronic illnesses such as diabetes, cardiovascular diseases, and mental health conditions continue to rise in prevalence, placing additional strain on already stressed health infrastructures.",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "Aveva, quindi, acclarato che la contribuente aveva acquistato i beni dalla capogruppo ad un prezzo lontano dal ‘valore normale’ di cui al citato art. 110, comma 7.La società impugnava, con tre distinti ricorsi, gli atti impositiviinnanzi alla Commissione tributaria provinciale di Firenze, che, previa riunione, li accoglieva: annullava il rilievo relativo al evidenziando che già era stata adottata, per altra annualità, una pronuncia favorevole alla contribuente, sulla base della non applicabilità della metodologia del TNMM e che l’Ufficio aveva errato nella scelta dei ‘comparabili’; annullava il rilievo relativo all’IVA in",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "Rate 2% of gross revenues from UK in-scope activities above threshold; however, taxpayers may apply an alternative calculation method calculated based on operating margin in respect of in-scope activities where they are loss- making or have a very low proﬁ t margin Thresholds £500m revenues from in-scope activities provided globally and £25m of revenue from in-scope activities provided to UK users per 12-month accounting period.",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "The UK DST applies from 1 April 2020 and is payable annually, nine months after relevant accounting period. The legislation is included in Finance Act 2020, which received Royal Assent on 22 July 2020.  The UK tax authority published its DST manual on 19 March 2020, which explains the structure and details of the UK DST.  This manual includes what is meant by digital services activity and revenue, deﬁ nitions of a user and identifying revenue of UK users, detail on the role and responsibilities of the responsible member, as well as further details of the administration and compliance framework that applies for DST.  There have been updates to the manual since its publication, including the deﬁ nition of online services, the compliance framework and the list of countries that have taxes that are considered to be similar to the UK DST for the purposes of cross-border relief. In March 2021, HMRC made further changes to the DST manual, introducing a section on the compliance framework, updating the guidance on submitting returns for groups with non-GBP consolidated accounts and adding Spain to its list of countries with similar DST (for which cross-border relief would be allowed). Further, from 14 June 2022, the Malaysian Service Tax on Digital Services by Foreign Service Providers is no longer considered by HMRC to be similar to the UK DST for the purposes of cross-border relief.  Any claims for cross-border tax relief made before 14 June 2022 will be honored, but no new claims for relief relating to this tax will be accepted. In August 2024, HMRC updated the DST manual to conﬁ rm that for the purposes of DST cross-border relief, HMRC considers the Canadian DST to be similar to the UK DST. Rate 2% of gross revenues from UK in-scope activities above threshold; however, taxpayers may apply an alternative calculation method calculated based on operating margin in respect of in-scope activities where they are loss- making or have a very low proﬁ t margin Thresholds £500m revenues from in-scope activities provided globally and £25m of revenue from in-scope activities provided to UK users per 12-month accounting period. The ﬁ rst £25m of revenues is not subject to the tax.  £500m and £25m thresholds are applied to total revenues arising to a group from in-scope activities, rather than on an activity-by-activity basis. The group upon which the thresholds are tested is determined by reference to accounting consolidation principles. Exclusions Provision of an online marketplace by a ﬁ nancial services provider where upwards of 50% of revenues relate to the creation/trading of ﬁ nancial assets Effective date 1 April 2020 Reference Links as below EY Global Tax Alerts US initiates review of other countries' imposition of DSTs on US companies and opens comment period on nonreciprocal trade arrangements (25 February 2025) Six country Joint Statement on transitional approach to existing unilateral measures during period before Pillar One is in effect (25 October 2021) USTR releases ﬁ ndings of Section 301 investigation on DST regimes of Austria, Spain and the UK, and 301 ﬁ ndings on Vietnam’s currency valuation practices | EY – Global (21 January 2021)  USTR proposes 25% punitive tariff on Austrian, Indian, Italian, Spanish, Turkish and UK origin goods in response to each country’s DST; Terminates investigations for Brazil, Czech Republic, EU and Indonesia | EY – Global (29 March 2021)  USTR initiates investigations into DSTs either adopted, or under consideration, by certain jurisdictions (4 June 2020) UK releases draft clauses and guidance on Digital Services Tax (12 July 2019) UK proposes Digital Services Tax: unilateral measure announced in Budget 2018 (5 November 2018) USTR announces 25% punitive tariffs on six speciﬁ c countries in response to their DSTs; Suspends tariffs for 180 days (4 June 2021) Status  The Finance Act, 2019 and the Companies Income Tax (Signiﬁ cant Economic Presence) Order, 2020 expanded the scope of taxation of non-resident companies (NRCs) performing digital services in Nigeria.    NRCs deriving income from digital services are deemed to derive income from Nigeria to the extent that such NRCs have a signiﬁ cant economic presence (SEP) in the country.  NRCs deemed to have a SEP in Nigeria are required to register for taxes and to comply with the relevant income tax ﬁ ling and payment obligations in Nigeria.  The Finance Act 2021 provided that non-resident companies liable to tax on proﬁ ts arising from digital goods and services under the SEP rule may be assessed on fair and reasonable percentage of turnover if there is no assessable proﬁ t, the assessable proﬁ t is less than expected or the assessable proﬁ t cannot be ascertained. Scope Foreign companies undertaking the following activities are deemed to have a SEP in Nigeria:  Category 1 – A foreign company using digital platforms to derive gross income equal to or greater than N25 million (or its equivalence in other currencies) in a year of assessment, from any of the following activities (or combination thereof):   Streaming, or downloading services of digital contents to any person in Nigeria  Transmission of data collected about Nigerian users, which has been generated from such user’s activities on a digital interface, including a website or mobile application.  Provision of goods or services directly or indirectly to Nigerians through digital platforms.  Provision of intermediation services through digital platforms that link suppliers and customers in Nigeria. Category 2 – A foreign company that uses a Nigerian domain name (.ng) or registers a website address in Nigeria.  Category 3 – A foreign company that has a purposeful and sustained interaction with persons in Nigeria by customizing its digital platform to target persons in Nigeria or reﬂ ecting the prices of its products, services or options of billing or payment in the local currency, Naira.  Rate Corporate income tax at 30% of taxable proﬁ ts.Thresholds N25 million (approximately US$26,000) for Category 1 transactions Exclusions Foreign companies covered under any multilateral/consensus agreement to address tax challenges arising from digitalization of the economy to which Nigeria is a party, to the extent that  such agreement is effective.  So far, Nigeria has not signed up for BEPS 2.0. Status  Currently Mexico does not impose a DST (December 2021).  Effective as of 2022, Mexico City has a contribution on deliveries (e.g., food, parcels) through digital platforms. The new tax is equal to 2% of the total charge before taxes for each delivery made through ﬁ xed or mobile devices that allow users to contract for the delivery of parcels, food, provisions, or any type of merchandise delivered in Mexico City’s territory. This tax is to be paid by the platform and cannot be transferred to the clients or persons making the delivery.  The tax authority (Servicio de Administración Tributaria) published the Miscellaneous Fiscal Resolution which includes rules and guidance on the remission of withholding tax by foreign digital service providers.  In 2018, a DST Bill was submitted to the Mexican Congress to apply a 3% tax on the revenue of digital providers that are residents in Mexico or that have a permanent establishment in the country. The Bill was not approved by the Congress.  As of March 2021, a 16% VAT is applicable on digital services provided by foreign residents with no permanent establishment in Mexico when the recipient of the service is located in Mexico. This tax applies to certain digital services such as providing access to content for users, gaming and learning; the law also applies to platforms providing intermediation services. The foreign digital supplier is obligated to meet several compliance and disclosure obligations before the Mexican tax authorities. These obligations include, but are not limited to, registering in Mexico, reporting and emitting tax on a monthly basis and providing certain disclosures as to services provided in Mexico  In January 2024, the tax authorities published a list of almost 201 foreign digital service providers registered before the Mexican tax authorities. Scope  Mexico City has a contribution on deliveries (e.g., food, parcels) through digital platforms for the delivery of parcels, food, provisions, or any type of merchandise delivered in the Mexico City territory. This tax is paid by the platform and cannot be transferred to the clients or persons making the delivery.  VAT is applicable on digital services provided by foreign residents with no permanent establishment in Mexico when the recipient of the service is located in Mexico. This tax applies to certain digital services such as providing access to content for users, gaming and learning; the law also applies to platforms providing intermediation services Rate  Mexico City contribution on deliveries 2%  VAT 16% Thresholds N/A Exclusions N/A",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "One of the most urgent international climate issues today is the accelerating rise in global temperatures, driven primarily by human activity. This warming trend—commonly referred to as global warming—is the result of increased concentrations of greenhouse gases such as carbon dioxide, methane, and nitrous oxide in the atmosphere. These gases trap heat, causing the Earths surface and oceans to warm at unprecedented rates",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "The Court formulated the following principle of law for the lower court to follow: in transfer pricing cases involving intra-group transfers of goods between two low-risk companies, where risk is reduced due to the existence of a single production centre essentially producing against confirmed orders, TNMM is more appropriate than CUP because profit margin is a more indicative criterion than nominal price. In such a model, nominal price is not the result of a free market.According to the Court, TNMM with an ROS indicator is fully acceptable and may be preferred to CUP in centrally controlled, low-risk distribution models where prices are not determined by free market forces.",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide.1. Overview of the Global Health Landscape In the past decades, the world has seen repeated reminders of how vulnerable health systems can be. Outbreaks of infectious diseases, such as influenza variants, coronaviruses, viral hemorrhagic fevers, and antimicrobial-resistant pathogens, have emphasized the need for robust surveillance systems. Climate-driven events—including heat waves, floods, and vector expansion—have intensified the burden of diseases such as malaria, dengue, and cholera. Simultaneously, chronic illnesses such as diabetes, cardiovascular diseases, and mental health conditions continue to rise in prevalence, placing additional strain on already stressed health infrastructures.",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        },
        {
            "doc_text": "Rate 2% of gross revenues from UK in-scope activities above threshold...",
            "label_text": "Contains a tax problem, tax solution and discussion about a tax topic or tax solution or tax type"
        }
    ]
}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Print response
print(response.status_code)
print(json.dumps(response.json(), indent=2))
